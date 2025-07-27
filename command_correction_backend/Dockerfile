# 使用 Python 官方镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-dev \
    gcc \
    flac \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y ffmpeg
# 升级 pip
RUN pip install --upgrade pip

# 复制依赖文件并安装依赖
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# 复制应用代码
COPY . .

EXPOSE 1228

# 启动应用
CMD ["python", "main.py"]
