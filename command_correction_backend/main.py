from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
import tempfile
import os
import sys
import inputClassify
import cmdClassify
import modify_area_extraction
from pydub import AudioSegment
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import configparser
import openai
from openai import OpenAI

# 建立 logs 資料夾
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("my_api_logger")
logger.setLevel(logging.INFO)
logger.handlers.clear()  # 清除之前所有 handlers

formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

file_handler = TimedRotatingFileHandler(
    filename="logs/server.log",
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8"
)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("✅ 測試 log 是否成功寫入")

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://eduact.csie.ncu.edu.tw"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  
    allow_methods=["*"],
    allow_headers=["*"],
)

current_path = os.path.dirname(os.path.abspath(__file__))
config = configparser.ConfigParser()
config.read(os.path.join(current_path, './config.ini'))

client_openai = OpenAI(
    organization = config['openai']['organization'],
    api_key = config['openai']['api_key']
)

input_classify = inputClassify.InputClassifier()
cmd_classify = cmdClassify.CMDClassifier()
modify_area = modify_area_extraction.Modify_area_extractor()



def force_print(*args):
    print(*args, flush=True)
    sys.stdout.flush()

@app.get("/test", response_class=PlainTextResponse)
async def test():
    return "Test"

@app.post("/input_classifier/")
async def input_classifier(data: dict):
    if 'content' not in data:
        logger.warning("No content provided in input_classifier")
        raise HTTPException(status_code=400, detail="No content provided")

    input_content = data['content']
    logger.info(f"Received input for classification: {input_content}")
    prediction = input_classify.predict(input_content)
    logger.info(f"Prediction result: {prediction}")
    return {'result': prediction}

@app.post("/input_classifier_llm/")
async def input_classifier_llm(data: dict):
    if 'content' not in data:
        logger.warning("No content provided in input_classifier")
        raise HTTPException(status_code=400, detail="No content provided")
    input_content = data['content']
    logger.info(f"Received input for classification: {input_content}")
    prompt = """
            你是一個語音輸入指令分類器，請根據以下規則判斷輸入的語句屬於哪一類，並**只輸出對應的代號（不要加上說明或其他文字）**：

            0：一般語音內容（如敘述、聊天內容）

            1：編輯指令（例如「請把某個字改成別的字」、「刪除某句話」）

            2：操作指令，若為此類請再細分為：
            2-1：停止錄音指令（例如：停止錄音、錄到這邊就好）
            2-2：送出目前內容指令（例如：送出、傳出去）
            2-3：重新開始錄音指令（例如：重新開始、繼續錄音）

            請根據輸入句子，**只輸出下列其中之一**：0、1、2-1、2-2 或 2-3。
            """
    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o-mini",  # 或 gpt-4、gpt-3.5-turbo 等
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_content}
            ]
        )
        result = response.choices[0].message.content.strip()
        logger.info(f"Prediction result: {result}")
        return {"result": result}
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise HTTPException(status_code=500, detail="OpenAI API call failed")


@app.post("/error_correction/")
async def error_correction(data: dict):
    if 'text' not in data or 'command' not in data:
        logger.warning("Missing text or command in error_correction")
        raise HTTPException(status_code=400, detail="No content provided")

    text = data['text']
    command = data['command']
    logger.info(f"Correction requested with text: '{text}', command: '{command}'")

    content = text
    command_type = cmd_classify.predict(command)
    logger.info(f"Command type predicted: {command_type}")
    instruction = text + '[SEP]' + command

    try:
        result = modify_area.predict(instruction)
        labels = result['labels']
        offsets = result['offsets']
        text = result['text']
        logger.info(f"Label prediction: {labels}")
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

    #content = text  # 預設為原始文本

    if 'B-Modify' in labels:
        modify_site = labels.index('B-Modify')
        modify_start, modify_end = offsets[modify_site]

        if command_type == 0:  # 替換
            if 'B-Filling' in labels:
                fill_site = labels.index('B-Filling')
                fill_start, fill_end = offsets[fill_site]
                fill_char = text[fill_start:fill_end]

                content = content[:modify_start] + fill_char + content[modify_start+1:]

        elif command_type == 1:  # 插入
            if 'B-Filling' in labels:
                fill_site = labels.index('B-Filling')
                fill_start, fill_end = offsets[fill_site]
                fill_char = text[fill_start:fill_end]

                content = content[:modify_start] + fill_char + content[modify_start:]

        elif command_type == 2:  # 刪除
            content = content[:modify_start] + content[modify_start+1:]

    logger.info(f"Final corrected content: {content}")

    return {
        'status': 'success',
        'corrected_text': content,
        'command_type': command_type,
    }