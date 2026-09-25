"""Worker agent: executes one assigned sub-task via NIM."""
from agency import nim_client, registry
from agency.agent_memory import AgentMemoryMixin


class WorkerAgent(AgentMemoryMixin):
    def __init__(self, role: str, memory=None, use_memory: bool = True):
        if role not in registry.AGENTS:
            raise ValueError(f"Unknown role {role!r}. Choose from {list(registry.AGENTS)}")
        self.role = role
        self.system = registry.AGENTS[role]["system"]
        self.model = registry.model_for(role)
        self._memory = memory
        self.use_memory = use_memory

    def _delegate(self, cls_name: str, module: str, instruction: str, context: str,
                  stream_output: bool, on_retry) -> dict:
        import importlib

        mod = importlib.import_module(module)
        agent = getattr(mod, cls_name)(model=self.model, memory=self._mem(), use_memory=self.use_memory)
        return agent.run(instruction, context=context, stream_output=stream_output, on_retry=on_retry)

    def run(self, instruction: str, context: str = "", stream_output: bool = False, on_retry=None) -> dict:
        if self.role == "scraper":
            return self._delegate("ScraperAgent", "agency.scraper", instruction, context, stream_output, on_retry)
        if self.role == "yt_scraper":
            return self._delegate("YTScraperAgent", "agency.yt_scraper", instruction, context, stream_output, on_retry)
        if self.role == "lead_gen":
            return self._delegate("LeadGenAgent", "agency.lead_gen", instruction, context, stream_output, on_retry)
        if self.role == "marketing":
            return self._delegate("MarketingAgent", "agency.marketing", instruction, context, stream_output, on_retry)
        if self.role == "url_data":
            return self._delegate("UrlDataAgent", "agency.url_data", instruction, context, stream_output, on_retry)
        if self.role == "trading":
            return self._delegate("TradingAgent", "agency.trading", instruction, context, stream_output, on_retry)
        if self.role == "image_maker":
            return self._delegate("ImageMakerAgent", "agency.image_maker", instruction, context, stream_output, on_retry)
        user = f"Context:\n{context}\n\nTask:\n{instruction}" if context else instruction
        snippet = self._recall(instruction)
        if snippet:
            user += f"\n\nRelevant past memory:\n{snippet}"
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user},
        ]
        # Workers: thinking OFF + smaller max_tokens to save 40 RPM budget
        res = nim_client.chat(
            messages=messages,
            model=self.model,
            temperature=0.7,
            top_p=0.95,
            max_tokens=4096,
            enable_thinking=False,
            stream_output=stream_output,
            on_retry=on_retry,
        )
        self._store(self.role, instruction, res["content"])
        return {"role": self.role, "model": self.model, "output": res["content"]}
