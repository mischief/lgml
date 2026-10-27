/*** Tensor graphs on ggml backends.
 * @module lgml
 */

/*
 * The systemd service protocol: notify messages, the file descriptor
 * store, and anonymous memory files to keep in it. Linux only; the
 * systemd build option leaves it out.
 */

#define _GNU_SOURCE
#include <sys/mman.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <errno.h>
#include <stddef.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <lua.h>
#include <lauxlib.h>

#include "lgml.h"

/* The most fds that one notify message carries. */
#define NOTIFY_MAXFDS	16

static int
push_errno(lua_State *L, const char *what)
{
	lua_pushnil(L);
	lua_pushfstring(L, "%s: %s", what, strerror(errno));
	return 2;
}

/* Send state with fds to $NOTIFY_SOCKET. 0, or -1 with errno. */
static int
notify(const char *state, size_t len, const int *fds, int nfds)
{
	union {
		struct cmsghdr	hdr;
		char		buf[CMSG_SPACE(sizeof(int) * NOTIFY_MAXFDS)];
	} cm;
	struct sockaddr_un sun;
	struct msghdr msg;
	struct iovec iov;
	struct cmsghdr *c;
	const char *path = getenv("NOTIFY_SOCKET");
	size_t plen;
	ssize_t n;
	int s, saved;

	plen = strlen(path);
	if (plen < 2 || plen >= sizeof(sun.sun_path) || (path[0] != '/' && path[0] != '@')) {
		errno = EINVAL;
		return -1;
	}
	memset(&sun, 0, sizeof(sun));
	sun.sun_family = AF_UNIX;
	memcpy(sun.sun_path, path, plen);
	/* An abstract socket name starts with a zero byte. */
	if (path[0] == '@')
		sun.sun_path[0] = '\0';
	if ((s = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0)) == -1)
		return -1;
	iov.iov_base = (void *)state;
	iov.iov_len = len;
	memset(&msg, 0, sizeof(msg));
	msg.msg_name = &sun;
	msg.msg_namelen = (socklen_t)(offsetof(struct sockaddr_un, sun_path) + plen);
	msg.msg_iov = &iov;
	msg.msg_iovlen = 1;
	if (nfds > 0) {
		memset(&cm, 0, sizeof(cm));
		msg.msg_control = cm.buf;
		msg.msg_controllen = CMSG_SPACE(sizeof(int) * nfds);
		c = CMSG_FIRSTHDR(&msg);
		c->cmsg_level = SOL_SOCKET;
		c->cmsg_type = SCM_RIGHTS;
		c->cmsg_len = CMSG_LEN(sizeof(int) * nfds);
		memcpy(CMSG_DATA(c), fds, sizeof(int) * nfds);
	}
	n = sendmsg(s, &msg, MSG_NOSIGNAL);
	saved = errno;
	close(s);
	errno = saved;
	return n == (ssize_t)len ? 0 : -1;
}

/*** Send a message to the service manager.
 * It does nothing outside a systemd service: without `NOTIFY_SOCKET`.
 * @tparam string state lines such as `"READY=1"` or
 * `"FDSTORE=1\nFDNAME=x"`
 * @tparam[opt] {int,...} fds file descriptors to send with it
 * @treturn[1] boolean true when sent, false without `NOTIFY_SOCKET`
 * @treturn[2] nil
 * @treturn[2] string the error
 * @function sd_notify
 */
static int
l_sd_notify(lua_State *L)
{
	size_t len;
	const char *state = luaL_checklstring(L, 1, &len);
	int fds[NOTIFY_MAXFDS], nfds = 0, i;
	lua_Integer fd;

	if (!lua_isnoneornil(L, 2)) {
		luaL_checktype(L, 2, LUA_TTABLE);
		nfds = (int)luaL_len(L, 2);
		luaL_argcheck(L, nfds <= NOTIFY_MAXFDS, 2, "too many fds");
		for (i = 0; i < nfds; i++) {
			lua_geti(L, 2, i + 1);
			fd = luaL_checkinteger(L, -1);
			luaL_argcheck(L, fd >= 0 && fd <= INT_MAX, 2, "bad fd");
			fds[i] = (int)fd;
			lua_pop(L, 1);
		}
	}
	if (getenv("NOTIFY_SOCKET") == NULL) {
		lua_pushboolean(L, 0);
		return 1;
	}
	if (notify(state, len, fds, nfds) == -1)
		return push_errno(L, "sd_notify");
	lua_pushboolean(L, 1);
	return 1;
}

/*** Wait until the service manager has read the earlier messages.
 * It sends `BARRIER=1` with a pipe and waits for the manager to close
 * it. Call it before exit, so the manager does not drop messages of a
 * process that is gone.
 * @tparam[opt=5000] int ms the most milliseconds to wait
 * @treturn[1] boolean true, or false without `NOTIFY_SOCKET`
 * @treturn[2] nil
 * @treturn[2] string the error, or `"timeout"`
 * @function sd_barrier
 */
static int
l_sd_barrier(lua_State *L)
{
	lua_Integer ms = luaL_optinteger(L, 1, 5000);
	struct pollfd pfd;
	int p[2], r;

	luaL_argcheck(L, ms >= 0 && ms <= INT_MAX, 1, "bad time");
	if (getenv("NOTIFY_SOCKET") == NULL) {
		lua_pushboolean(L, 0);
		return 1;
	}
	if (pipe(p) == -1)
		return push_errno(L, "sd_barrier");
	(void)fcntl(p[0], F_SETFD, FD_CLOEXEC);
	(void)fcntl(p[1], F_SETFD, FD_CLOEXEC);
	r = notify("BARRIER=1", 9, &p[1], 1);
	close(p[1]);
	if (r == -1) {
		r = errno;
		close(p[0]);
		errno = r;
		return push_errno(L, "sd_barrier");
	}
	pfd.fd = p[0];
	pfd.events = 0;
	do
		r = poll(&pfd, 1, (int)ms);
	while (r == -1 && errno == EINTR);
	close(p[0]);
	if (r == -1)
		return push_errno(L, "sd_barrier");
	if (r == 0) {
		lua_pushnil(L);
		lua_pushliteral(L, "timeout");
		return 2;
	}
	lua_pushboolean(L, 1);
	return 1;
}

/*** Take the file descriptors that the service manager passed.
 * They come from socket units and the file descriptor store. It reads
 * `LISTEN_PID`, `LISTEN_FDS` and `LISTEN_FDNAMES`, sets close-on-exec
 * on each fd, and removes the three variables, so a second call and
 * child processes get none.
 * @treturn {table,...} `{ fd, name }` entries, in order; name is
 * `"unknown"` when the manager gave none
 * @function sd_listen_fds
 */
static int
l_sd_listen_fds(lua_State *L)
{
	const char *spid = getenv("LISTEN_PID"), *sn = getenv("LISTEN_FDS");
	const char *names = getenv("LISTEN_FDNAMES"), *p, *e;
	char *end;
	long pid, n, i;

	lua_newtable(L);
	if (spid == NULL || sn == NULL)
		goto out;
	errno = 0;
	pid = strtol(spid, &end, 10);
	if (errno != 0 || *end != '\0' || pid != (long)getpid())
		goto out;
	n = strtol(sn, &end, 10);
	if (errno != 0 || *end != '\0' || n <= 0 || n > INT_MAX - 3)
		goto out;
	p = names;
	for (i = 0; i < n; i++) {
		(void)fcntl(3 + (int)i, F_SETFD, FD_CLOEXEC);
		lua_createtable(L, 0, 2);
		lua_pushinteger(L, 3 + i);
		lua_setfield(L, -2, "fd");
		if (p != NULL && *p != '\0') {
			e = strchr(p, ':');
			lua_pushlstring(L, p, e ? (size_t)(e - p) : strlen(p));
			p = e ? e + 1 : NULL;
		} else
			lua_pushliteral(L, "unknown");
		lua_setfield(L, -2, "name");
		lua_seti(L, -2, i + 1);
	}
out:
	unsetenv("LISTEN_PID");
	unsetenv("LISTEN_FDS");
	unsetenv("LISTEN_FDNAMES");
	return 1;
}

/*** Make an anonymous memory file, with close-on-exec.
 * @tparam string name the name, for /proc and debugging
 * @treturn[1] int the fd
 * @treturn[2] nil
 * @treturn[2] string the error
 * @function memfd
 */
static int
l_memfd(lua_State *L)
{
	const char *name = luaL_checkstring(L, 1);
	int fd = memfd_create(name, MFD_CLOEXEC);

	if (fd == -1)
		return push_errno(L, "memfd");
	lua_pushinteger(L, fd);
	return 1;
}

void
lgml_open_sd(lua_State *L)
{
	static const luaL_Reg fns[] = {
		{ "sd_notify", l_sd_notify },
		{ "sd_barrier", l_sd_barrier },
		{ "sd_listen_fds", l_sd_listen_fds },
		{ "memfd", l_memfd },
		{ NULL, NULL }
	};

	luaL_setfuncs(L, fns, 0);
}
