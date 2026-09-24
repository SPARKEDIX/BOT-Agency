"""Worker agent: executes one assigned sub-task via NIM."""
from agency import nim_client, registry


class WorkerAgent:
    def __init__(self, role: str):
        if role not in registry.AGENTS:
            raise ValueError(f"Unknown role {role!r}. Choose from {list(registry.AGENTS)}")
        self.role = role
        self.system = registry.AGENTS[role]["system"]
        self.model = registry.model_for(role)

    def run(self, instruction: str, context: str = "", stream_output: bool = False, on_retry=None) -> dict:
        if self.role == "scraper":
            from agency.scraper import ScraperAgent

            return ScraperAgent(model=self.model).run(
                instruction, context=context, stream_output=stream_output, on_retry=on_retry
            )
        if self.role == "image_maker":
            from agency.image_maker import ImageMakerAgent

            return ImageMakerAgent().run(
                instruction, context=context, stream_output=stream_output, on_retry=on_retry
            )
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": f"Context:\n{context}\n\nTask:\n{instruction}" if context else instruction},
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
        return {"role": self.role, "model": self.model, "output": res["content"]}
