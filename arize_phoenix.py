from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

def init_phoenix():
    tracer_provider = register()
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)