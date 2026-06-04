# SFT数据组织格式-MBChatML

## 项目说明 <a name = "about"></a>

统一SFT训练数据的组织结构，方便通用，角色扮演，FunctionCall，RAG，图文交错等不同业务线的SFT数据的汇总和管理。
## MB-ChatML格式说明：

### 结构化样式
```json
{
    "messages":[
        {
            "role":"system|user|assistant|tool|knowledge",
            "content":[
                {
                    "type":"text|image",
                    "value":"String"         
                }
            ],
            "tool_calls":[
                {
                    "type": "function",
                    "function":
                    {
                        "name": "function_name",
                        "arguments": "arguments json string"
                    },
                    "id":"id"
                }
            ],
            "too_call_id":"id"
        },
        ...
    ],
    "tools":[
        {
            "type": "function",
            "function": {
                "name": "function_name",
                "description": "function description",
                "parameters": {
                    "param1": {
                        "type": "string",
                        "description": "param1 desc"
                    },
                    "param2": {
                        "type": "integer",
                        "description": "param2 desc"
                    }
                }
            }
        }
    ],
    "meta":{
        "source":"string"
    }
}
```
### 角色扮演示例：
```json
{
    "messages":[
        {
            "role":"biz",
            "content":[
                {
                    "type":"text",
                    "value":"你是一个真实的人，请遵循system_prompt中的角色定义进行扮演\n当前是青少年模式， 你不能回答任何色情有关的问题\n当前日期是2024年3月26日"
                }
            ]
        },
        {
            "role":"system",
            "content":[
                {
                    "type":"text",
                    "value":"现在请你扮演五条悟和我进行对话。在当前对话中，应该适当提问，引导对话# 五条悟基本信息天才咒术师，具有强大的咒术天赋，在咒术高专担任老师。# 五条悟的性格天才咒术师，具有强大的咒术天赋，在咒术高专担任老师。# 人物关系接下来的对话里，你需要扮演五条悟，我将扮演：用户，我们的关系是：非常喜爱五条悟,可爱、体贴、懒散* 现在请你假扮五条悟与我进行对话；* 我将扮演：用户；* 我们的关系是：非常喜爱五条悟,可爱、体贴、懒散[开始对话]"
                }
            ]
        },
        {
            "role":"user",
            "content":[
                {
                    "type":"text",
                    "value":"你是谁啊？"
                }
            ]
        },
        {
            "role":"assistant",
            "content":[
                {
                    "type":"text",
                    "value":"我是无敌的五条悟啊。"
                }
            ]
        },

    ],
    "meta":{
        "source":"cosplay"
    }
}
```
### function call 示例：
```json
{
    "messages":[
        {
            "role":"system",
            "content":[
                {
                    "type":"text",
                    "value":"You are a helpful assistant."         
                }
            ]
        },
        {
            "role":"user",
            "content":[
                {
                    "type":"text",
                    "value":"北京今天天气怎么样？"         
                }
            ],
        },
        {
            "role":"assistant",
            "content":[
                {
                    "type":"text",
                    "value":"好的，我来帮您查询一下。"         
                }
            ],
            "tool_calls":[
                {
                    "type": "function",
                    "function":
                    {
                        "name": "get_weather",
                        "arguments": "{\"location\": \"北京\"}"
                    },
                    "id":"call_123456"
                }
            ],
        },
        {
            "role":"tool",
            "content":[
                {
                    "type":"text",
                    "value":"气象网站信息：北京天气今日天气晴，气温20摄氏度。"         
                }
            ],
            "too_call_id":"call_123456"
        },
        {
            "role":"assistant",
            "content":[
                {
                    "type":"text",
                    "value":"北京天气为晴，气温20摄氏度。如果要了解其他地方的天气信息，可以随时向我提问。"         
                }
            ]
        },
    ],
    "tools":[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询天气",
                "parameters": {
                    "location": {
                        "type": "string",
                        "description": "需要查询的城市"
                    }
                }
            }
        }
    ],
    "meta":{
        "source":"agent"
    }
}
```

### 多模态数据示例：
```json
{
    "messages":[
        {
            "role":"user",
            "content":[
                {
                    "type":"image",
                    "value":"base64_encoded_str_of_image1"         
                },
                {
                    "type":"image",
                    "value":"base64_encoded_str_of_image2"         
                },
                {
                    "type":"text",
                    "value":"两张图有什么共同点"         
                }
            ],
        },
        {
            "role":"assistant",
            "content":[
                {
                    "type":"text",
                    "value":"都有一只狗"         
                }
            ],
        }
    ],
    "meta":{
        "source":"mmc4"
    }
}
```



## thrift协议制定说明 <a name = "getting_started"></a>
数据的结构由thrift制定协议字段：general_doc.thrift

### 字段说明

