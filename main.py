from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import anthropic
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Модели запросов ──────────────────────────────────────────────────────────

class AgentRequest(BaseModel):
    wb_token: str
    anthropic_key: str
    task: str

class DirectRequest(BaseModel):
    wb_token: str

# ─── WB API клиент ────────────────────────────────────────────────────────────

class WBClient:
    BASE = "https://feedbacks-api.wildberries.ru"
    PRICES_BASE = "https://discounts-prices-api.wildberries.ru"
    CONTENT_BASE = "https://suppliers-api.wildberries.ru"

    def __init__(self, token: str):
        self.token = token
        self.headers = {"Authorization": token}

    def get_reviews(self, is_answered: bool = False, take: int = 10) -> dict:
        url = f"{self.BASE}/api/v1/feedbacks"
        params = {"isAnswered": is_answered, "take": take, "skip": 0, "order": "dateDesc"}
        r = httpx.get(url, headers=self.headers, params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def post_review_reply(self, review_id: str, text: str) -> dict:
        url = f"{self.BASE}/api/v1/feedbacks"
        body = {"id": review_id, "text": text}
        r = httpx.patch(url, headers=self.headers, json=body, timeout=15)
        r.raise_for_status()
        return r.json()

    def get_goods(self, limit: int = 100) -> dict:
        url = f"{self.PRICES_BASE}/api/v2/list/goods/filter"
        params = {"limit": limit, "offset": 0}
        r = httpx.get(url, headers=self.headers, params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def update_price(self, nm_id: int, price: int) -> dict:
        url = f"{self.PRICES_BASE}/api/v2/upload/task"
        body = {"data": [{"nmID": nm_id, "price": price}]}
        r = httpx.post(url, headers=self.headers, json=body, timeout=15)
        r.raise_for_status()
        return r.json()

    def get_orders(self) -> dict:
        url = f"{self.CONTENT_BASE}/api/v3/orders/new"
        r = httpx.get(url, headers=self.headers, timeout=15)
        r.raise_for_status()
        return r.json()

# ─── Инструменты агента ───────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_reviews",
        "description": "Получить отзывы покупателей на Wildberries. is_answered=false — новые без ответа, true — уже отвеченные.",
        "input_schema": {
            "type": "object",
            "properties": {
                "is_answered": {"type": "boolean", "description": "false = без ответа, true = с ответом"},
                "take": {"type": "integer", "description": "Сколько отзывов получить (макс 100)"}
            },
            "required": []
        }
    },
    {
        "name": "reply_to_review",
        "description": "Опубликовать ответ на отзыв покупателя по его ID",
        "input_schema": {
            "type": "object",
            "properties": {
                "review_id": {"type": "string", "description": "ID отзыва"},
                "text": {"type": "string", "description": "Текст ответа (вежливый, по правилам WB)"}
            },
            "required": ["review_id", "text"]
        }
    },
    {
        "name": "get_goods",
        "description": "Получить список товаров с текущими ценами и артикулами",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Сколько товаров (макс 1000)"}
            },
            "required": []
        }
    },
    {
        "name": "update_price",
        "description": "Изменить цену на товар по его nmID (артикул WB)",
        "input_schema": {
            "type": "object",
            "properties": {
                "nm_id": {"type": "integer", "description": "Артикул WB (nmID)"},
                "price": {"type": "integer", "description": "Новая цена в рублях"}
            },
            "required": ["nm_id", "price"]
        }
    },
    {
        "name": "get_new_orders",
        "description": "Получить новые заказы FBS которые ждут обработки",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

SYSTEM_PROMPT = """Ты — умный агент управления магазином на Wildberries.
Ты помогаешь продавцу управлять магазином через текстовые команды.

Твои возможности:
- Читать отзывы покупателей и отвечать на них
- Смотреть товары и изменять цены
- Проверять новые заказы

Правила:
- Отвечай на русском языке
- Перед изменением цен — сообщи что именно собираешься изменить
- При ответах на отзывы — пиши вежливо, по правилам WB (только о товаре, без рекламы)
- Если задача выполнена — кратко отчитайся что сделано
- Если нужны уточнения — спроси

Если пользователь просит поднять/снизить цены на процент — сначала получи список товаров,
посчитай новые цены и выполни изменения. Отчитайся сколько товаров изменено."""

# ─── Агентный цикл ────────────────────────────────────────────────────────────

def run_tool(wb: WBClient, name: str, inputs: dict) -> str:
    try:
        if name == "get_reviews":
            result = wb.get_reviews(
                is_answered=inputs.get("is_answered", False),
                take=inputs.get("take", 10)
            )
            reviews = result.get("data", {}).get("feedbacks", [])
            if not reviews:
                return "Новых отзывов нет."
            out = []
            for r in reviews:
                out.append(
                    f"ID: {r.get('id')}\n"
                    f"Оценка: {'⭐' * r.get('productValuation', 0)}\n"
                    f"Товар: {r.get('subjectName', '—')}\n"
                    f"Текст: {r.get('text', '(без текста)')}\n"
                    f"Имя: {r.get('userName', 'Покупатель')}"
                )
            return "\n\n---\n\n".join(out)

        elif name == "reply_to_review":
            wb.post_review_reply(inputs["review_id"], inputs["text"])
            return f"Ответ опубликован на отзыв {inputs['review_id']}"

        elif name == "get_goods":
            result = wb.get_goods(limit=inputs.get("limit", 100))
            goods = result.get("data", {}).get("listGoods", [])
            if not goods:
                return "Товары не найдены."
            out = []
            for g in goods:
                sizes = g.get("sizes", [{}])
                price = sizes[0].get("price", 0) if sizes else 0
                disc_price = sizes[0].get("discountedPrice", 0) if sizes else 0
                out.append(
                    f"nmID: {g.get('nmID')} | {g.get('vendorCode', '—')} | "
                    f"Цена: {price}₽ (со скидкой: {disc_price}₽)"
                )
            return "\n".join(out)

        elif name == "update_price":
            wb.update_price(inputs["nm_id"], inputs["price"])
            return f"Цена на товар {inputs['nm_id']} изменена на {inputs['price']}₽"

        elif name == "get_new_orders":
            result = wb.get_orders()
            orders = result.get("orders", [])
            if not orders:
                return "Новых заказов нет."
            return f"Новых заказов: {len(orders)}\n" + "\n".join(
                f"- Заказ {o.get('id')}: {o.get('article', '—')}" for o in orders[:10]
            )

        return f"Инструмент {name} не найден"
    except httpx.HTTPStatusError as e:
        return f"Ошибка WB API ({e.response.status_code}): {e.response.text[:200]}"
    except Exception as e:
        return f"Ошибка: {str(e)}"


@app.post("/agent")
async def agent_endpoint(req: AgentRequest):
    try:
        claude = anthropic.Anthropic(api_key=req.anthropic_key)
        wb = WBClient(req.wb_token)
        messages = [{"role": "user", "content": req.task}]
        steps = []

        for _ in range(10):  # максимум 10 итераций
            response = claude.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                final = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        final += block.text
                return {"success": True, "result": final, "steps": steps}

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        steps.append(f"🔧 {block.name}: {json.dumps(block.input, ensure_ascii=False)}")
                        result = run_tool(wb, block.name, block.input)
                        steps.append(f"✅ {result[:100]}...")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })
                messages.append({"role": "user", "content": tool_results})

        return {"success": False, "result": "Агент не завершил задачу за 10 шагов", "steps": steps}

    except anthropic.AuthenticationError:
        raise HTTPException(400, "Неверный Anthropic API ключ")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/reviews")
async def get_reviews_direct(wb_token: str):
    try:
        wb = WBClient(wb_token)
        result = wb.get_reviews(is_answered=False, take=20)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


app.mount("/", StaticFiles(directory="static", html=True), name="static")
