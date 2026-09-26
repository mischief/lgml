-- Cases from llama.cpp tests/test-json-schema-to-grammar.cpp.
-- Each case: name, ok (expected success), schema text, grammar.
return {
	{
		name = "min 0",
		ok = true,
		schema = [[{
            "type": "integer",
            "minimum": 0
        }]],
		grammar = [[

            root ::= ([0] | [1-9] [0-9]{0,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "min 1",
		ok = true,
		schema = [[{
            "type": "integer",
            "minimum": 1
        }]],
		grammar = [[

            root ::= ([1-9] [0-9]{0,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "min 3",
		ok = true,
		schema = [[{
            "type": "integer",
            "minimum": 3
        }]],
		grammar = [[

            root ::= ([1-2] [0-9]{1,15} | [3-9] [0-9]{0,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "min 9",
		ok = true,
		schema = [[{
            "type": "integer",
            "minimum": 9
        }]],
		grammar = [[

            root ::= ([1-8] [0-9]{1,15} | [9] [0-9]{0,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "min 10",
		ok = true,
		schema = [[{
            "type": "integer",
            "minimum": 10
        }]],
		grammar = [[

            root ::= ([1] ([0-9]{1,15}) | [2-9] [0-9]{1,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "min 25",
		ok = true,
		schema = [[{
            "type": "integer",
            "minimum": 25
        }]],
		grammar = [[

            root ::= ([1] [0-9]{2,15} | [2] ([0-4] [0-9]{1,14} | [5-9] [0-9]{0,14}) | [3-9] [0-9]{1,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "max 30",
		ok = true,
		schema = [[{
            "type": "integer",
            "maximum": 30
        }]],
		grammar = [[

            root ::= ("-" [1-9] [0-9]{0,15} | [0-9] | ([1-2] [0-9] | [3] "0"))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "min -5",
		ok = true,
		schema = [[{
            "type": "integer",
            "minimum": -5
        }]],
		grammar = [[

            root ::= ("-" ([0-5]) | [0] | [1-9] [0-9]{0,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "min -123",
		ok = true,
		schema = [[{
            "type": "integer",
            "minimum": -123
        }]],
		grammar = [[

            root ::= ("-" ([0-9] | ([1-8] [0-9] | [9] [0-9]) | "1" ([0-1] [0-9] | [2] [0-3])) | [0] | [1-9] [0-9]{0,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "max -5",
		ok = true,
		schema = [[{
            "type": "integer",
            "maximum": -5
        }]],
		grammar = [[

            root ::= ("-" ([0-4] [0-9]{1,15} | [5-9] [0-9]{0,15}))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "max 1",
		ok = true,
		schema = [[{
            "type": "integer",
            "maximum": 1
        }]],
		grammar = [[

            root ::= ("-" [1-9] [0-9]{0,15} | [0-1])
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "max 100",
		ok = true,
		schema = [[{
            "type": "integer",
            "maximum": 100
        }]],
		grammar = [[

            root ::= ("-" [1-9] [0-9]{0,15} | [0-9] | ([1-8] [0-9] | [9] [0-9]) | "100")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "min 0 max 23",
		ok = true,
		schema = [[{
            "type": "integer",
            "minimum": 0,
            "maximum": 23
        }]],
		grammar = [[

            root ::= ([0-9] | ([1] [0-9] | [2] [0-3]))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "min 15 max 300",
		ok = true,
		schema = [[{
            "type": "integer",
            "minimum": 15,
            "maximum": 300
        }]],
		grammar = [[

            root ::= (([1] ([5-9]) | [2-9] [0-9]) | ([1-2] [0-9]{2} | [3] "00"))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "min 5 max 30",
		ok = true,
		schema = [[{
            "type": "integer",
            "minimum": 5,
            "maximum": 30
        }]],
		grammar = [[

            root ::= ([5-9] | ([1-2] [0-9] | [3] "0"))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "min -123 max 42",
		ok = true,
		schema = [[{
            "type": "integer",
            "minimum": -123,
            "maximum": 42
        }]],
		grammar = [[

            root ::= ("-" ([0-9] | ([1-8] [0-9] | [9] [0-9]) | "1" ([0-1] [0-9] | [2] [0-3])) | [0-9] | ([1-3] [0-9] | [4] [0-2]))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "min -10 max 10",
		ok = true,
		schema = [[{
            "type": "integer",
            "minimum": -10,
            "maximum": 10
        }]],
		grammar = [[

            root ::= ("-" ([0-9] | "10") | [0-9] | "10")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "unknown type",
		ok = false,
		schema = [[{
            "type": "kaboom"
        }]],
		grammar = [[]],
	},
	{
		name = "invalid type",
		ok = false,
		schema = [[{
            "type": 123
        }]],
		grammar = [[]],
	},
	{
		name = "empty schema (object)",
		ok = true,
		schema = [[{}]],
		grammar = [[

            array ::= "[" space ( value ("," space value)* )? space "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
            root ::= object
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            value ::= object | array | string | number | boolean | null
        ]],
	},
	{
		name = "exotic formats",
		ok = true,
		schema = [[{
            "items": [
                { "format": "date" },
                { "format": "uuid" },
                { "format": "time" },
                { "format": "date-time" }
            ]
        }]],
		grammar = [[

            date ::= [0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )
            date-string ::= "\"" date "\""
            date-time ::= date "T" time
            date-time-string ::= "\"" date-time "\""
            root ::= "[" space tuple-0 "," space uuid "," space tuple-2 "," space tuple-3 space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            time ::= ([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )
            time-string ::= "\"" time "\""
            tuple-0 ::= date-string
            tuple-2 ::= time-string
            tuple-3 ::= date-time-string
            uuid ::= "\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\""
        ]],
	},
	{
		name = "string",
		ok = true,
		schema = [[{
            "type": "string"
        }]],
		grammar = [[

            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char* "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "string w/ min length 1",
		ok = true,
		schema = [[{
            "type": "string",
            "minLength": 1
        }]],
		grammar = [[

            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char+ "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "string w/ min length 3",
		ok = true,
		schema = [[{
            "type": "string",
            "minLength": 3
        }]],
		grammar = [[

            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char{3,} "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "string w/ max length",
		ok = true,
		schema = [[{
            "type": "string",
            "maxLength": 3
        }]],
		grammar = [[

            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char{0,3} "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "string w/ min & max length",
		ok = true,
		schema = [[{
            "type": "string",
            "minLength": 1,
            "maxLength": 4
        }]],
		grammar = [[

            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char{1,4} "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "boolean",
		ok = true,
		schema = [[{
            "type": "boolean"
        }]],
		grammar = [[

            root ::= ("true" | "false")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "integer",
		ok = true,
		schema = [[{
            "type": "integer"
        }]],
		grammar = [[

            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= ("-"? integral-part)
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "string const",
		ok = true,
		schema = [[{
            "const": "foo"
        }]],
		grammar = [[

            root ::= "\"foo\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "non-string const",
		ok = true,
		schema = [[{
            "const": 123
        }]],
		grammar = [[

            root ::= "123"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "non-string enum",
		ok = true,
		schema = [=[{
            "enum": ["red", "amber", "green", null, 42, ["foo"]]
        }]=],
		grammar = [[

            root ::= ("\"red\"" | "\"amber\"" | "\"green\"" | "null" | "42" | "[\"foo\"]")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "string array",
		ok = true,
		schema = [[{
            "type": "array",
            "prefixItems": { "type": "string" }
        }]],
		grammar = [[

            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "[" space (string ("," space string)*)? space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        ]],
	},
	{
		name = "nullable string array",
		ok = true,
		schema = [[{
            "type": ["array", "null"],
            "prefixItems": { "type": "string" }
        }]],
		grammar = [[

            alternative-0 ::= "[" space (string ("," space string)*)? space "]"
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            null ::= "null"
            root ::= alternative-0 | null
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        ]],
	},
	{
		name = "tuple1",
		ok = true,
		schema = [[{
            "prefixItems": [{ "type": "string" }]
        }]],
		grammar = [[

            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "[" space string space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        ]],
	},
	{
		name = "tuple2",
		ok = true,
		schema = [[{
            "prefixItems": [{ "type": "string" }, { "type": "number" }]
        }]],
		grammar = [[

            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "[" space string "," space number space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        ]],
	},
	{
		name = "array with empty items",
		ok = true,
		schema = [[{
            "type": "array",
            "items": {}
        }]],
		grammar = [[

            array ::= "[" space ( value ("," space value)* )? space "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            item ::= object
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
            root ::= "[" space (item ("," space item)*)? space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            value ::= object | array | string | number | boolean | null
        ]],
	},
	{
		name = "array with empty items and prefixItems",
		ok = true,
		schema = [[{
            "type": "array",
            "items": {},
            "prefixItems": { "type": "string" }
        }]],
		grammar = [[

            array ::= "[" space ( value ("," space value)* )? space "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            item ::= object
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
            root ::= "[" space (item ("," space item)*)? space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            value ::= object | array | string | number | boolean | null
        ]],
	},
	{
		name = "number",
		ok = true,
		schema = [[{
            "type": "number"
        }]],
		grammar = [[

            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "minItems",
		ok = true,
		schema = [[{
            "items": {
                "type": "boolean"
            },
            "minItems": 2
        }]],
		grammar = [[

            boolean ::= ("true" | "false")
            root ::= "[" space boolean ("," space boolean)+ space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "maxItems 0",
		ok = true,
		schema = [[{
            "items": {
                "type": "boolean"
            },
            "maxItems": 0
        }]],
		grammar = [[

            boolean ::= ("true" | "false")
            root ::= "[" space  space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "maxItems 1",
		ok = true,
		schema = [[{
            "items": {
                "type": "boolean"
            },
            "maxItems": 1
        }]],
		grammar = [[

            boolean ::= ("true" | "false")
            root ::= "[" space boolean? space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "maxItems 2",
		ok = true,
		schema = [[{
            "items": {
                "type": "boolean"
            },
            "maxItems": 2
        }]],
		grammar = [[

            boolean ::= ("true" | "false")
            root ::= "[" space (boolean ("," space boolean)?)? space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "min + maxItems",
		ok = true,
		schema = [[{
            "items": {
                "type": ["number", "integer"]
            },
            "minItems": 3,
            "maxItems": 5
        }]],
		grammar = [[

            decimal-part ::= [0-9]{1,16}
            integer ::= ("-"? integral-part)
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            item ::= number | integer
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "[" space item ("," space item){2,4} space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "min + max items with min + max values across zero",
		ok = true,
		schema = [[{
            "items": {
                "type": "integer",
                "minimum": -12,
                "maximum": 207
            },
            "minItems": 3,
            "maxItems": 5
        }]],
		grammar = [[

            item ::= ("-" ([0-9] | "1" [0-2]) | [0-9] | ([1-8] [0-9] | [9] [0-9]) | ([1] [0-9]{2} | [2] "0" [0-7]))
            root ::= "[" space item ("," space item){2,4} space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "min + max items with min + max values",
		ok = true,
		schema = [[{
            "items": {
                "type": "integer",
                "minimum": 12,
                "maximum": 207
            },
            "minItems": 3,
            "maxItems": 5
        }]],
		grammar = [[

            item ::= (([1] ([2-9]) | [2-9] [0-9]) | ([1] [0-9]{2} | [2] "0" [0-7]))
            root ::= "[" space item ("," space item){2,4} space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "simple regexp",
		ok = true,
		schema = [[{
            "type": "string",
            "pattern": "^abc?d*efg+(hij)?kl$"
        }]],
		grammar = [[

            root ::= "\"" ("ab" "c"? "d"* "ef" "g"+ ("hij")? "kl") "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "regexp escapes",
		ok = true,
		schema = [[{
            "type": "string",
            "pattern": "^\\[\\]\\{\\}\\(\\)\\|\\+\\*\\?$"
        }]],
		grammar = [[

            root ::= "\"" ("[]{}()|+*?") "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "regexp quote",
		ok = true,
		schema = [[{
            "type": "string",
            "pattern": "^\"$"
        }]],
		grammar = [[

            root ::= "\"" ("\"") "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "regexp with top-level alternation",
		ok = true,
		schema = [[{
            "type": "string",
            "pattern": "^A|B|C|D$"
        }]],
		grammar = [[

            root ::= "\"" ("A" | "B" | "C" | "D") "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "regexp",
		ok = true,
		schema = [[{
            "type": "string",
            "pattern": "^(\\([0-9]{1,3}\\))?[0-9]{3}-[0-9]{4} a{3,5}nd...$"
        }]],
		grammar = [[

            dot ::= [^\x0A\x0D]
            root ::= "\"" (("(" root-1{1,3} ")")? root-1{3,3} "-" root-1{4,4} " " "a"{3,5} "nd" dot dot dot) "\""
            root-1 ::= [0-9]
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "required props in original order",
		ok = true,
		schema = [[{
            "type": "object",
            "properties": {
                "b": {"type": "string"},
                "c": {"type": "string"},
                "a": {"type": "string"}
            },
            "required": [
                "a",
                "b",
                "c"
            ],
            "additionalProperties": false,
            "definitions": {}
        }]],
		grammar = [[

            a-kv ::= "\"a\"" space ":" space string
            b-kv ::= "\"b\"" space ":" space string
            c-kv ::= "\"c\"" space ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "{" space b-kv "," space c-kv "," space a-kv space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        ]],
	},
	{
		name = "1 optional prop",
		ok = true,
		schema = [[{
            "properties": {
                "a": {
                "type": "string"
                }
            },
            "additionalProperties": false
        }]],
		grammar = [[

            a-kv ::= "\"a\"" space ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "{" space  (a-kv )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        ]],
	},
	{
		name = "N optional props",
		ok = true,
		schema = [[{
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
                "c": {"type": "string"}
            },
            "additionalProperties": false
        }]],
		grammar = [[

            a-kv ::= "\"a\"" space ":" space string
            a-rest ::= ( "," space b-kv )? b-rest
            b-kv ::= "\"b\"" space ":" space string
            b-rest ::= ( "," space c-kv )?
            c-kv ::= "\"c\"" space ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "{" space  (a-kv a-rest | b-kv b-rest | c-kv )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        ]],
	},
	{
		name = "required + optional props each in original order",
		ok = true,
		schema = [[{
            "properties": {
                "b": {"type": "string"},
                "a": {"type": "string"},
                "d": {"type": "string"},
                "c": {"type": "string"}
            },
            "required": ["a", "b"],
            "additionalProperties": false
        }]],
		grammar = [[

            a-kv ::= "\"a\"" space ":" space string
            b-kv ::= "\"b\"" space ":" space string
            c-kv ::= "\"c\"" space ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            d-kv ::= "\"d\"" space ":" space string
            d-rest ::= ( "," space c-kv )?
            root ::= "{" space b-kv "," space a-kv ( "," space ( d-kv d-rest | c-kv ) )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        ]],
	},
	{
		name = "additional props",
		ok = true,
		schema = [[{
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "number"}}
        }]],
		grammar = [[

            additional-kv ::= string ":" space additional-value
            additional-value ::= "[" space (number ("," space number)*)? space "]"
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "{" space  (additional-kv ( "," space additional-kv )* )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        ]],
	},
	{
		name = "additional props (true)",
		ok = true,
		schema = [[{
            "type": "object",
            "additionalProperties": true
        }]],
		grammar = [[

            array ::= "[" space ( value ("," space value)* )? space "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
            root ::= object
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            value ::= object | array | string | number | boolean | null
        ]],
	},
	{
		name = "additional props (implicit)",
		ok = true,
		schema = [[{
            "type": "object"
        }]],
		grammar = [[

            array ::= "[" space ( value ("," space value)* )? space "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
            root ::= object
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            value ::= object | array | string | number | boolean | null
        ]],
	},
	{
		name = "empty w/o additional props",
		ok = true,
		schema = [[{
            "type": "object",
            "additionalProperties": false
        }]],
		grammar = [[

            root ::= "{" space  space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "required + additional props",
		ok = true,
		schema = [[{
            "type": "object",
            "properties": {
                "a": {"type": "number"}
            },
            "required": ["a"],
            "additionalProperties": {"type": "string"}
        }]],
		grammar = [[

            a-kv ::= "\"a\"" space ":" space number
            additional-k ::= ["] ( [a] char+ | [^"a] char* )? ["]
            additional-kv ::= additional-k ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "{" space a-kv ( "," space ( additional-kv ( "," space additional-kv )* ) )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        ]],
	},
	{
		name = "optional + additional props",
		ok = true,
		schema = [[{
            "type": "object",
            "properties": {
                "a": {"type": "number"}
            },
            "additionalProperties": {"type": "number"}
        }]],
		grammar = [[

            a-kv ::= "\"a\"" space ":" space number
            a-rest ::= ( "," space additional-kv )*
            additional-k ::= ["] ( [a] char+ | [^"a] char* )? ["]
            additional-kv ::= additional-k ":" space number
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "{" space  (a-kv a-rest | additional-kv ( "," space additional-kv )* )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "required + optional + additional props",
		ok = true,
		schema = [[{
            "type": "object",
            "properties": {
                "and": {"type": "number"},
                "also": {"type": "number"}
            },
            "required": ["and"],
            "additionalProperties": {"type": "number"}
        }]],
		grammar = [[

            additional-k ::= ["] ( [a] ([l] ([s] ([o] char+ | [^"o] char*) | [^"s] char*) | [n] ([d] char+ | [^"d] char*) | [^"ln] char*) | [^"a] char* )? ["]
            additional-kv ::= additional-k ":" space number
            also-kv ::= "\"also\"" space ":" space number
            also-rest ::= ( "," space additional-kv )*
            and-kv ::= "\"and\"" space ":" space number
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "{" space and-kv ( "," space ( also-kv also-rest | additional-kv ( "," space additional-kv )* ) )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "optional props with empty name",
		ok = true,
		schema = [[{
            "properties": {
                "": {"type": "integer"},
                "a": {"type": "integer"}
            },
            "additionalProperties": {"type": "integer"}
        }]],
		grammar = [[

            -kv ::= "\"\"" space ":" space root
            -rest ::= ( "," space a-kv )? a-rest
            a-kv ::= "\"a\"" space ":" space integer
            a-rest ::= ( "," space additional-kv )*
            additional-k ::= ["] ( [a] char+ | [^"a] char* ) ["]
            additional-kv ::= additional-k ":" space integer
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            integer ::= ("-"? integral-part)
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= ("-"? integral-part)
            root0 ::= "{" space  (-kv -rest | a-kv a-rest | additional-kv ( "," space additional-kv )* )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "optional props with nested names",
		ok = true,
		schema = [[{
            "properties": {
                "a": {"type": "integer"},
                "aa": {"type": "integer"}
            },
            "additionalProperties": {"type": "integer"}
        }]],
		grammar = [[

            a-kv ::= "\"a\"" space ":" space integer
            a-rest ::= ( "," space aa-kv )? aa-rest
            aa-kv ::= "\"aa\"" space ":" space integer
            aa-rest ::= ( "," space additional-kv )*
            additional-k ::= ["] ( [a] ([a] char+ | [^"a] char*) | [^"a] char* )? ["]
            additional-kv ::= additional-k ":" space integer
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            integer ::= ("-"? integral-part)
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= "{" space  (a-kv a-rest | aa-kv aa-rest | additional-kv ( "," space additional-kv )* )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "optional props with common prefix",
		ok = true,
		schema = [[{
            "properties": {
                "ab": {"type": "integer"},
                "ac": {"type": "integer"}
            },
            "additionalProperties": {"type": "integer"}
        }]],
		grammar = [[

            ab-kv ::= "\"ab\"" space ":" space integer
            ab-rest ::= ( "," space ac-kv )? ac-rest
            ac-kv ::= "\"ac\"" space ":" space integer
            ac-rest ::= ( "," space additional-kv )*
            additional-k ::= ["] ( [a] ([b] char+ | [c] char+ | [^"bc] char*) | [^"a] char* )? ["]
            additional-kv ::= additional-k ":" space integer
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            integer ::= ("-"? integral-part)
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= "{" space  (ab-kv ab-rest | ac-kv ac-rest | additional-kv ( "," space additional-kv )* )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "top-level $ref",
		ok = true,
		schema = [[{
            "$ref": "#/definitions/foo",
            "definitions": {
                "foo": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "a"
                    ],
                    "additionalProperties": false
                }
            }
        }]],
		grammar = [[

            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            ref-definitions-foo ::= "{" space ref-definitions-foo-a-kv space "}"
            ref-definitions-foo-a-kv ::= "\"a\"" space ":" space string
            root ::= ref-definitions-foo
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        ]],
	},
	{
		name = "anyOf",
		ok = true,
		schema = [[{
            "anyOf": [
                {"$ref": "#/definitions/foo"},
                {"$ref": "#/definitions/bar"}
            ],
            "definitions": {
                "foo": {
                    "properties": {"a": {"type": "number"}}
                },
                "bar": {
                    "properties": {"b": {"type": "number"}}
                }
            },
            "type": "object"
        }]],
		grammar = [[

            alternative-0 ::= ref-definitions-foo
            alternative-1 ::= ref-definitions-bar
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            ref-definitions-bar ::= "{" space  (ref-definitions-bar-b-kv )? space "}"
            ref-definitions-bar-b-kv ::= "\"b\"" space ":" space number
            ref-definitions-foo ::= "{" space  (ref-definitions-foo-a-kv )? space "}"
            ref-definitions-foo-a-kv ::= "\"a\"" space ":" space number
            root ::= alternative-0 | alternative-1
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "anyOf $ref",
		ok = true,
		schema = [[{
            "properties": {
                "a": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "number"}
                    ]
                },
                "b": {
                    "anyOf": [
                        {"$ref": "#/properties/a/anyOf/0"},
                        {"type": "boolean"}
                    ]
                }
            },
            "type": "object"
        }]],
		grammar = [[

            a ::= string | number
            a-kv ::= "\"a\"" space ":" space a
            a-rest ::= ( "," space b-kv )?
            b ::= b-0 | boolean
            b-0 ::= string
            b-kv ::= "\"b\"" space ":" space b
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "{" space  (a-kv a-rest | b-kv )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        ]],
	},
	{
		name = "mix of allOf, anyOf and $ref (similar to https://json.schemastore.org/tsconfig.json)",
		ok = true,
		schema = [[{
            "allOf": [
                {"$ref": "#/definitions/foo"},
                {"$ref": "#/definitions/bar"},
                {
                "anyOf": [
                    {"$ref": "#/definitions/baz"},
                    {"$ref": "#/definitions/bam"}
                ]
                }
            ],
            "definitions": {
                "foo": {
                    "properties": {"a": {"type": "number"}}
                },
                "bar": {
                    "properties": {"b": {"type": "number"}}
                },
                "bam": {
                    "properties": {"c": {"type": "number"}}
                },
                "baz": {
                    "properties": {"d": {"type": "number"}}
                }
            },
            "type": "object"
        }]],
		grammar = [[

            a-kv ::= "\"a\"" space ":" space number
            b-kv ::= "\"b\"" space ":" space number
            c-kv ::= "\"c\"" space ":" space number
            d-kv ::= "\"d\"" space ":" space number
            d-rest ::= ( "," space c-kv )?
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "{" space a-kv "," space b-kv ( "," space ( d-kv d-rest | c-kv ) )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "allOf with enum schema",
		ok = true,
		schema = [[{
            "allOf": [
                {"$ref": "#/definitions/foo"}
            ],
            "definitions": {
                "foo": {
                    "type": "string",
                    "enum": ["a", "b"]
                }
            }
        }]],
		grammar = [[

            root ::= ("\"a\"" | "\"b\"")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "allOf with multiple enum schemas",
		ok = true,
		schema = [[{
            "allOf": [
                {"$ref": "#/definitions/foo"},
                {"$ref": "#/definitions/bar"}
            ],
            "definitions": {
                "foo": {
                    "type": "string",
                    "enum": ["a", "b", "c"]
                },
                "bar": {
                    "type": "string",
                    "enum": ["b", "c", "d"]
                }
            }
        }]],
		grammar = [[

            root ::= ("\"b\"" | "\"c\"")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "conflicting names",
		ok = true,
		schema = [[{
            "type": "object",
            "properties": {
                "number": {
                "type": "object",
                "properties": {
                    "number": {
                    "type": "object",
                        "properties": {
                            "root": {
                                "type": "number"
                            }
                        },
                        "required": [
                            "root"
                        ],
                        "additionalProperties": false
                    }
                },
                "required": [
                    "number"
                ],
                "additionalProperties": false
                }
            },
            "required": [
                "number"
            ],
            "additionalProperties": false,
            "definitions": {}
        }]],
		grammar = [[

            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            number- ::= "{" space number-number-kv space "}"
            number-kv ::= "\"number\"" space ":" space number-
            number-number ::= "{" space number-number-root-kv space "}"
            number-number-kv ::= "\"number\"" space ":" space number-number
            number-number-root-kv ::= "\"root\"" space ":" space number
            root ::= "{" space number-kv space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "description only (no type) treated as unconstrained",
		ok = true,
		schema = [[{"description": "The 0-based index of the last line to be retrieved (inclusive). If None, read until the end of the file."}]],
		grammar = [[

            array ::= "[" space ( value ("," space value)* )? space "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
            root ::= value
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            value ::= object | array | string | number | boolean | null
        ]],
	},
	{
		name = "literal string with escapes",
		ok = true,
		schema = [[{
            "properties": {
                "code": {
                    "const": " \r \n \" \\ ",
                    "description": "Generated code",
                    "title": "Code",
                    "type": "string"
                }
            },
            "required": [
                "code"
            ],
            "title": "DecoderResponse",
            "type": "object"
        }]],
		grammar = [[

            code ::= "\" \\r \\n \\\" \\\\ \""
            code-kv ::= "\"code\"" space ":" space code
            root ::= "{" space code-kv space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        ]],
	},
	{
		name = "regexp with non-capturing group",
		ok = true,
		schema = [[{
                "type": "string",
                "pattern": "^(?:foo|bar)baz$"
            }]],
		grammar = [[

                root ::= "\"" (("foo" | "bar") "baz") "\""
                space ::= | " " | "\n"{1,2} [ \t]{0,20}
            ]],
	},
	{
		name = "regexp with nested non-capturing groups",
		ok = true,
		schema = [[{
                "type": "string",
                "pattern": "^(?:(?:ab)+c)?d$"
            }]],
		grammar = [[

                root ::= "\"" ((("ab")+ "c")? "d") "\""
                space ::= | " " | "\n"{1,2} [ \t]{0,20}
            ]],
	},
}
