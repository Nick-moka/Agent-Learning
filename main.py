
# ==================== 第三方库 ====================

# LangChain 社区扩展：SentenceTransformer 嵌入模型
# 用于将文本转换为向量表示，支持多种预训练模型（如 all-MiniLM-L6-v2）
# 安装: pip install langchain-community sentence-transformers
from langchain_community.embeddings import SentenceTransformerEmbeddings

# LangChain 社区扩展：Chroma 向量数据库
# 轻量级本地向量数据库，用于存储和检索文档的向量嵌入
# 支持相似度搜索，是 RAG（检索增强生成）的核心组件
# 安装: pip install langchain-community chromadb
from langchain_community.vectorstores import Chroma

# ==================== 自定义工具模块 ====================

# 从 utils/file_read.py 导入自定义文件读取和处理函数
# loadFile: 读取各种格式的文件（如 PDF, Word, TXT 等）
# split_text: 将长文本切分为适合向量化的文本块（chunk）
from utils.readFile import readFile,splitContent,load_folder_files

# ==================== Python 标准库 ====================

# 操作系统接口模块
# 用于处理文件路径、环境变量、目录操作等系统级功能
import os

# ==================== 阿里云 DashScope SDK ====================

# 阿里云 DashScope 核心 SDK
# 用于调用通义千问（Qwen）系列大模型 API
# 安装: pip install dashscope
import dashscope

# 从 dashscope 中导入文本生成模块
# Generation 类用于调用大模型进行文本生成对话
# 使用前需要设置 API Key: dashscope.api_key = "your-api-key"
from dashscope import Generation

# ==================== FastAPI Web 框架 ====================

# FastAPI 核心组件
# 用于构建高性能 RESTful API
# app = FastAPI() 创建应用实例
# @app.get("/path") 装饰器定义路由
# 安装: pip install fastapi uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import FileResponse, Response #0625 网页每次会获取.ico文件，增加处理依赖
from fastapi.security import APIKeyHeader
from fastapi import Security

# Pydantic 数据验证模型
# BaseModel 用于定义请求/响应数据的结构（Schema）
# 自动进行类型校验和序列化/反序列化
# 安装: pip install pydantic（FastAPI 依赖，通常已安装）
from pydantic import BaseModel

# FastAPI 跨域资源共享中间件
# 允许前端应用（不同源）调用后端 API
# 配置 CORS 策略：允许哪些域名、方法、请求头
# 安装: pip install fastapi（已包含）
from fastapi.middleware.cors import CORSMiddleware

# ==================== Python 标准库 ====================

# 日志记录模块（Python 内置，无需安装）
# 用于记录程序运行信息、调试信息、错误日志
# 可配置日志级别、输出格式、输出位置（控制台/文件）
# 使用: logger = logging.getLogger(__name__)
import logging


from slowapi import Limiter,_rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

#redis限流
from core.rate_limit import limiter,LIMIT_RAG_CHAT,LIMIT_UPLOAD,LIMIT_HEALTH
from fastapi import FastAPI,Request,HTTPException
from fastapi.middleware.cors import CORSMiddleware

from starlette.responses import JSONResponse


from dotenv import load_dotenv
# 加载.env文件所有位置
load_dotenv()

#========从环境变量读取配置，不再硬编码=======
os.environ["HF_ENDPOINT"] = str(os.getenv("HF_MIRROR"))
pervir = os.getenv("PERSIST_DIR")
dashscope.api_key = str(os.getenv("QWEN_API_KEY"))
emdedding = SentenceTransformerEmbeddings(model_name = str(os.getenv("EMBED_MODEL")))
#0625  自定义接口密钥（调用方必须携带，防止恶意刷接口）
API_SECRET_KEY = os.getenv("API_SECRET_KEY")
CHUNSIZE = int(os.getenv("CHUNK_SIZE"))
CHUNKSIZE_OVERLOP = int(os.getenv("CHUNK_OVERLAP"))
top_k = int(os.getenv("TOP_K"))
DOC_FOLDER = os.getenv("DOC_FILE_PATH")


#告诉Swagger这个接口需要X-API-Key头
api_key_scheme = APIKeyHeader(name="X-API-Key",auto_error=False)

#日志配置，记录请求，报错信息
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s-%(levelname)s-%(message)s",
                    handlers=[logging.FileHandler("api_log.log",encoding="utf-8"),logging.StreamHandler()]
                    )
logger = logging.getLogger(__name__)

#初始化FastAPI实例
app = FastAPI(title="私有知识库RAG接口服务")
#0625 加固1:跨域中间件(允许网页，小程序，客户端跨域请求)=====
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"], #上线后改成指定域名，不要用*
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# #限流器方法封装
# limiter = Limiter(key_func=get_remote_address)
# #绑定到app
# app.state.limiter = limiter
# app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

#0629 redis限流
@app.middleware("http")
async def global_rate_limit_middleware(request:Request,call_next):
    # 白名单：健康检测接口，跳过鉴权、跳过限流
    url_path = request.url.path
    # 白名单：健康检查、自动接口文档，免密钥、免限流
    white_list = ["/health", "/docs", "/redoc", "/openapi.json"]
    if url_path in white_list:
        response = await call_next(request)
        return response
    #1.读取请求头的api密钥进行基础鉴权
    api_key = request.headers.get("X-API-KEY")
    if not api_key:
        return JSONResponse(status_code=401, content={"detail": "缺失接口访问密钥X-API-KEY"})
    print(f"今天是个好日子{url_path}")
    #2.根据接口路径匹配响应限流规则
    if url_path.startswith("/api/rag/chat"):
        current_rule = LIMIT_RAG_CHAT
    elif url_path.startswith("/api/doc/upload"):
        current_rule = LIMIT_UPLOAD
    else:
        current_rule = LIMIT_HEALTH
    #3.限流校验，捕获redis运行时异常做降级
    try:
        #hit(规则，业务命名空间，限流唯一标识)
        is_allow = limiter.hit(current_rule,"rag_service",api_key)
    except Exception as error:
        print(f"Redis限流读写异常:f{str(error)}")
        is_allow = True
    if not is_allow:
        return JSONResponse(status_code=429, content={"detail": "请求过于频繁，请稍后重试111"})
    response = await call_next(request)
    return response



# 请求体格式约束
class QuestionBody(BaseModel):
    question: str

#0625加固2：密钥鉴权依赖函数=======
def verify_secret_key(request:Request,api_key:str = Security(api_key_scheme)):
    if not api_key or api_key != API_SECRET_KEY:
        logger.warning("非法密钥访问，已拦截")
        raise HTTPException(status_code=404,detail="密钥错误，禁止访问")
    return True


#创建DB库
def getDB():
    if os.path.exists(pervir):
        return Chroma(embedding_function=emdedding,persist_directory=pervir)
    else:
        return None
    
#初始化数据
def buildVentor(filePath:str,chunSize=CHUNSIZE,overlop=CHUNKSIZE_OVERLOP):
    db = getDB()
    if db is not None:
        return db
    else:
        #读取文件中全部文本
        text_content = load_folder_files(filePath)
        if  not text_content.strip():
            logger.warning("文件夹内无有效文件")
            return None
        splitTexts = splitContent(text_content,chunSize = chunSize,overlop=overlop)
        db = Chroma.from_texts(texts= splitTexts,embedding=emdedding,persist_directory=pervir)
        db.persist()#持久化
        logger.info(f"批量入库完成，分片总数:{len(splitTexts)}")
        return db
#查找数据
def search_knowledge(question:str, topk:int=80)->str:
    db = getDB()
    if db is not None:
        resultList = db.similarity_search(query=question,k=3)
        context = ""
        for idx,item in enumerate(resultList):
            context += f"片段{idx}:{item.page_content}\n"
        prompt = f"""
你是企业内部知识库问答助手，严格遵守以下规则：
1. 仅根据下面【参考资料】内容回答用户问题，禁止编造、不能输出资料以外的信息；
2. 如果参考资料没有对应答案，直接回复：暂无相关资料；
3. 回答简洁清晰，分点说明。
【参考资料】{context}"""
        print(f"参考资料{context}")
        messageS = [{"role":"system","content":prompt},{"role":"user","content":question}]
        response = Generation.call(messages=messageS,top_k=topk,model="qwen-turbo",result_format="message")
        if response.status_code == 200:
            resultList = response.output.choices[0].message.content
            return f"请求的结果{resultList}"
            # for item in resultList:
            #     print(f"查找的结果:{item}\n")
        else:
            return f"请求结果出错,{response.status_code}"
    else:
        return "向量库未创建"
# ============ 添加 favicon 路由 ============
@app.get("/health")
def health_check():
    return{"code":200,"msg":"服务运行正常"}



@app.get("/favicon.ico")
async def get_favicon():
    # 方法1：如果 favicon.ico 在项目根目录
    favicon_path = "web.ico"
    
    # 方法2：使用绝对路径（更安全）
    # favicon_path = os.path.join(os.path.dirname(__file__), "favicon.ico")
    
    # 检查文件是否存在
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    else:
        # 如果文件不存在，返回 204（不报错）
        return Response(status_code=204)

#对外接口:POST请求
@app.post("/api/rag/chat",dependencies=[Depends(verify_secret_key)])
#limit限流 @limiter.limit("2/minute")
def rag_api(request:Request,body:QuestionBody):
    question = body.question.strip()#去空数据
    if len(question) < 1 or len(question) > 500:
        raise HTTPException(status_code=400,detail="问题长度需要控制在1~500字符之间")
    logger.info(f"收到用户提问：{question}")
    ans = search_knowledge(body.question,topk=top_k)
    return{
        "code":"200",
        "msg":"success",
        "data":{
            "answer":ans
        }
    }

@app.post("/api/rag/rebuild",dependencies=[Depends(verify_secret_key)])
# @limiter.limit("2/minute")
def rebuild_knowledge(request:Request):
    try:
        if os.path.exists(pervir):
            import shutil
            # #先断开Chroma连接再删
            # del globals()["app"]
            # gc.collect()
            shutil.rmtree(pervir)
        buildVentor(DOC_FOLDER)
        return{"code":"200","msg":"知识库重建完成"}
    except Exception as e:
        logger.error(f"重建失败:{str(e)}")
        raise HTTPException(status_code=500,detail=f"重建异常:{str(e)}")


#初始化创建一次向量库
# buildVentor(r"C:\Users\jess_jin\Desktop\moka\Agent\learning.txt",chunSize=500)
if __name__ == "__main__":
    # try:
    #     #项目启动自动加载的知识库
    #     buildVentor(DOC_FOLDER)
    # except Exception as e:
    #     logger.warning(f"知识库初始化跳过: {e}")



    import uvicorn
    uvicorn.run("main:app",host="0.0.0.0",port=8000)
    # context = search_knowledge("Agent")

    

    