# WorldCup Engine v1.2 — Docker Image
FROM python:3.11-slim

LABEL maintainer="WorldCup Engine"
LABEL version="1.2"
LABEL description="2026 World Cup prediction engine — Monte Carlo + ML + Arbitrage"

WORKDIR /app

# 仅安装计算依赖（无 LLM API 调用，纯 Python 计算）
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        numpy>=1.26 \
        scipy>=1.11 \
        pandas>=2.0 \
        requests>=2.31

# 复制工程文件
COPY code/ /app/code/
COPY data/ /app/data/

# 暴露 Web 端口
EXPOSE 8088

# 默认入口：启动 Web 仪表盘
ENTRYPOINT ["python3"]
CMD ["code/web/server.py", "8088"]

# 其他可用入口（通过 docker run 覆盖 CMD）：
# docker run -p 8088:8088 worldcup-engine
# docker run worldcup-engine code/run_tournament.py 100000
# docker run worldcup-engine code/models/calibrator.py report
# docker run worldcup-engine code/backtest/run_backtest.py 2018
