from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx
import anthropic
import json

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Модели ───────────────────────────────────────────────────────────────────

class AgentRequest(BaseModel):
    wb_token: str
    anthropic_key: str
    task: str

class GenerateReplyRequest(BaseModel):
    anthropic_key: str
    review_id: str
    stars: int
    product: str
    text: str
    user_name: str

class PublishReplyRequest(BaseModel):
    wb_token: str
    review_id: str
    text: str

class GenerateAllRequest(BaseModel):
    anthropic_key: str
    reviews: list

# ─── WB API клиент ────────────────────────────────────────────────────────────

class WBClient:
    FEEDBACK   = "https://feedbacks-api.wildberries.ru"
    PRICES     = "https://discounts-prices-api.wildberries.ru"
    SUPPLIERS  = "https://suppliers-api.wildberries.ru"
    ADVERT     = "https://advert-api.wildberries.ru"
    ANALYTICS  = "https://seller-analytics-api.wildberries.ru"
    CONTENT    = "https://content-api.wildberries.ru"
    MARKETPLACE= "https://marketplace-api.wildberries.ru"

    def __init__(self, token: str):
        self.h = {"Authorization": token}

    def get(self, base, path, params=None):
        r = httpx.get(f"{base}{path}", headers=self.h, params=params, timeout=20)
        r.raise_for_status(); return r.json()

    def post(self, base, path, body=None):
        r = httpx.post(f"{base}{path}", headers=self.h, json=body, timeout=20)
        r.raise_for_status(); return r.json()

    def patch(self, base, path, body=None):
        r = httpx.patch(f"{base}{path}", headers=self.h, json=body, timeout=20)
        r.raise_for_status(); return r.json()

    def put(self, base, path, body=None):
        r = httpx.put(f"{base}{path}", headers=self.h, json=body, timeout=20)
        r.raise_for_status(); return r.json()

    def delete(self, base, path, params=None):
        r = httpx.delete(f"{base}{path}", headers=self.h, params=params, timeout=20)
        r.raise_for_status(); return r.json()

    # ── Отзывы ────────────────────────────────────────────────────────────────
    def get_reviews(self, is_answered=False, take=20):
        return self.get(self.FEEDBACK, "/api/v1/feedbacks",
            {"isAnswered": is_answered, "take": take, "skip": 0, "order": "dateDesc"})

    def reply_review(self, review_id, text):
        return self.patch(self.FEEDBACK, "/api/v1/feedbacks", {"id": review_id, "text": text})

    def get_review_count(self):
        return self.get(self.FEEDBACK, "/api/v1/feedbacks/count")

    # ── Вопросы ───────────────────────────────────────────────────────────────
    def get_questions(self, is_answered=False, take=20):
        return self.get(self.FEEDBACK, "/api/v1/questions",
            {"isAnswered": is_answered, "take": take, "skip": 0, "order": "dateDesc"})

    def reply_question(self, question_id, text):
        return self.patch(self.FEEDBACK, "/api/v1/questions", {"id": question_id, "text": text})

    # ── Цены и скидки ─────────────────────────────────────────────────────────
    def get_goods(self, limit=100, offset=0):
        return self.get(self.PRICES, "/api/v2/list/goods/filter", {"limit": limit, "offset": offset})

    def update_prices(self, items: list):
        # items: [{"nmID": int, "price": int, "discount": int}, ...]
        return self.post(self.PRICES, "/api/v2/upload/task", {"data": items})

    def get_discount_history(self, nm_id: int):
        return self.get(self.PRICES, "/api/v2/history/goods/task", {"nmID": nm_id})

    # ── Заказы FBS ────────────────────────────────────────────────────────────
    def get_new_orders(self):
        return self.get(self.SUPPLIERS, "/api/v3/orders/new")

    def get_orders(self, date_from: str, limit=100):
        return self.get(self.SUPPLIERS, "/api/v3/orders",
            {"dateFrom": date_from, "limit": limit, "offset": 0})

    def cancel_order(self, order_id: int):
        return self.patch(self.SUPPLIERS, f"/api/v3/orders/{order_id}/cancel")

    # ── Поставки ──────────────────────────────────────────────────────────────
    def get_supplies(self, limit=10):
        return self.get(self.SUPPLIERS, "/api/v3/supplies", {"limit": limit, "offset": 0})

    def create_supply(self, name: str):
        return self.post(self.SUPPLIERS, "/api/v3/supplies", {"name": name})

    def add_order_to_supply(self, supply_id: str, order_id: int):
        return self.patch(self.SUPPLIERS, f"/api/v3/supplies/{supply_id}/orders/{order_id}")

    def close_supply(self, supply_id: str):
        return self.patch(self.SUPPLIERS, f"/api/v3/supplies/{supply_id}/close")

    def get_supply_orders(self, supply_id: str):
        return self.get(self.SUPPLIERS, f"/api/v3/supplies/{supply_id}/orders")

    def get_supply_barcode(self, supply_id: str):
        return self.get(self.SUPPLIERS, f"/api/v3/supplies/{supply_id}/barcode", {"type": "pdf"})

    # ── Склад / Остатки ───────────────────────────────────────────────────────
    def get_warehouses(self):
        return self.get(self.SUPPLIERS, "/api/v3/warehouses")

    def get_stocks(self, warehouse_id: int, skus: list = None):
        body = {"skus": skus or []}
        r = httpx.post(f"{self.MARKETPLACE}/api/v3/warehouses/{warehouse_id}/stocks",
            headers=self.h, json=body, timeout=20)
        r.raise_for_status(); return r.json()

    def update_stocks(self, warehouse_id: int, stocks: list):
        return self.put(self.MARKETPLACE, f"/api/v3/warehouses/{warehouse_id}/stocks",
            {"stocks": stocks})

    # ── Реклама ───────────────────────────────────────────────────────────────
    def get_adverts(self, status: int = None):
        params = {}
        if status: params["status"] = status
        return self.get(self.ADVERT, "/adv/v1/promotion/adverts", params)

    def get_advert_stat(self, advert_id: int, date_from: str, date_to: str):
        return self.post(self.ADVERT, "/adv/v2/fullstats",
            [{"id": advert_id, "dates": [date_from, date_to]}])

    def start_advert(self, advert_id: int):
        return self.get(self.ADVERT, f"/adv/v0/start", {"id": advert_id})

    def pause_advert(self, advert_id: int):
        return self.get(self.ADVERT, f"/adv/v0/pause", {"id": advert_id})

    def set_advert_budget(self, advert_id: int, amount: int):
        return self.post(self.ADVERT, "/adv/v1/budget/deposit",
            {"id": advert_id, "sum": amount, "type": 1, "return": False})

    def set_advert_cpm(self, advert_id: int, cpm: int, param: int, instrument: int = 8):
        return self.post(self.ADVERT, "/adv/v0/cpm",
            {"advertId": advert_id, "type": instrument, "cpm": cpm, "param": param})

    # ── Карточки товаров ──────────────────────────────────────────────────────
    def get_cards(self, limit=10, cursor=None):
        body = {"settings": {"cursor": {"limit": limit}, "filter": {"withPhoto": -1}}}
        if cursor: body["settings"]["cursor"]["updatedAt"] = cursor
        return self.post(self.CONTENT, "/content/v2/get/cards/list", body)

    def update_card(self, card: dict):
        return self.post(self.CONTENT, "/content/v2/cards/update", [card])

    def get_card_by_nm(self, nm_id: int):
        return self.post(self.CONTENT, "/content/v2/get/cards/list",
            {"settings": {"cursor": {"limit": 1},
             "filter": {"withPhoto": -1, "nmID": nm_id}}})

    # ── Аналитика ─────────────────────────────────────────────────────────────
    def get_sales_report(self, date_from: str, date_to: str):
        return self.get(self.ANALYTICS, "/api/v1/supplier/reportDetailByPeriod",
            {"dateFrom": date_from, "dateTo": date_to, "rrdid": 0, "limit": 100000})

    def get_paid_storage(self, date_from: str, date_to: str):
        return self.post(self.ANALYTICS, "/api/v1/paid_storage",
            {"dateFrom": date_from, "dateTo": date_to})


# ─── Инструменты агента ───────────────────────────────────────────────────────

TOOLS = [
    # Отзывы
    {"name": "get_reviews", "description": "Получить отзывы покупателей. is_answered=false — без ответа",
     "input_schema": {"type": "object", "properties": {
         "is_answered": {"type": "boolean"}, "take": {"type": "integer"}}, "required": []}},
    {"name": "reply_to_review", "description": "Опубликовать ответ на отзыв",
     "input_schema": {"type": "object", "properties": {
         "review_id": {"type": "string"}, "text": {"type": "string"}}, "required": ["review_id", "text"]}},
    {"name": "get_review_count", "description": "Узнать сколько отзывов ожидают ответа",
     "input_schema": {"type": "object", "properties": {}, "required": []}},

    # Вопросы
    {"name": "get_questions", "description": "Получить вопросы покупателей",
     "input_schema": {"type": "object", "properties": {
         "is_answered": {"type": "boolean"}, "take": {"type": "integer"}}, "required": []}},
    {"name": "reply_to_question", "description": "Ответить на вопрос покупателя",
     "input_schema": {"type": "object", "properties": {
         "question_id": {"type": "string"}, "text": {"type": "string"}}, "required": ["question_id", "text"]}},

    # Цены
    {"name": "get_goods", "description": "Получить товары с ценами",
     "input_schema": {"type": "object", "properties": {"limit": {"type": "integer"}}, "required": []}},
    {"name": "update_prices", "description": "Изменить цены и скидки на несколько товаров сразу. items = [{nmID, price, discount}]",
     "input_schema": {"type": "object", "properties": {
         "items": {"type": "array", "items": {"type": "object", "properties": {
             "nmID": {"type": "integer"}, "price": {"type": "integer"},
             "discount": {"type": "integer"}}}}}, "required": ["items"]}},

    # Заказы
    {"name": "get_new_orders", "description": "Получить новые заказы FBS",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "get_orders", "description": "Получить заказы с определённой даты (YYYY-MM-DD)",
     "input_schema": {"type": "object", "properties": {
         "date_from": {"type": "string"}}, "required": ["date_from"]}},
    {"name": "cancel_order", "description": "Отменить заказ по ID",
     "input_schema": {"type": "object", "properties": {
         "order_id": {"type": "integer"}}, "required": ["order_id"]}},

    # Поставки
    {"name": "get_supplies", "description": "Получить список поставок",
     "input_schema": {"type": "object", "properties": {"limit": {"type": "integer"}}, "required": []}},
    {"name": "create_supply", "description": "Создать новую поставку",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "add_order_to_supply", "description": "Добавить заказ в поставку",
     "input_schema": {"type": "object", "properties": {
         "supply_id": {"type": "string"}, "order_id": {"type": "integer"}},
         "required": ["supply_id", "order_id"]}},
    {"name": "close_supply", "description": "Закрыть поставку (передать в доставку)",
     "input_schema": {"type": "object", "properties": {
         "supply_id": {"type": "string"}}, "required": ["supply_id"]}},
    {"name": "get_supply_orders", "description": "Получить заказы в поставке",
     "input_schema": {"type": "object", "properties": {
         "supply_id": {"type": "string"}}, "required": ["supply_id"]}},

    # Склад
    {"name": "get_warehouses", "description": "Получить список складов продавца",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "update_stocks", "description": "Обновить остатки на складе. stocks = [{sku, amount}]",
     "input_schema": {"type": "object", "properties": {
         "warehouse_id": {"type": "integer"},
         "stocks": {"type": "array", "items": {"type": "object"}}},
         "required": ["warehouse_id", "stocks"]}},

    # Реклама
    {"name": "get_adverts", "description": "Получить список рекламных кампаний. status: 4=готова, 7=завершена, 8=отказ, 9=активна, 11=пауза",
     "input_schema": {"type": "object", "properties": {"status": {"type": "integer"}}, "required": []}},
    {"name": "get_advert_stat", "description": "Статистика рекламной кампании за период",
     "input_schema": {"type": "object", "properties": {
         "advert_id": {"type": "integer"},
         "date_from": {"type": "string"}, "date_to": {"type": "string"}},
         "required": ["advert_id", "date_from", "date_to"]}},
    {"name": "start_advert", "description": "Запустить рекламную кампанию",
     "input_schema": {"type": "object", "properties": {
         "advert_id": {"type": "integer"}}, "required": ["advert_id"]}},
    {"name": "pause_advert", "description": "Поставить рекламную кампанию на паузу",
     "input_schema": {"type": "object", "properties": {
         "advert_id": {"type": "integer"}}, "required": ["advert_id"]}},
    {"name": "set_advert_budget", "description": "Пополнить бюджет рекламной кампании",
     "input_schema": {"type": "object", "properties": {
         "advert_id": {"type": "integer"}, "amount": {"type": "integer"}},
         "required": ["advert_id", "amount"]}},
    {"name": "set_advert_cpm", "description": "Изменить ставку CPM в рекламной кампании",
     "input_schema": {"type": "object", "properties": {
         "advert_id": {"type": "integer"}, "cpm": {"type": "integer"},
         "param": {"type": "integer"}}, "required": ["advert_id", "cpm", "param"]}},

    # Карточки
    {"name": "get_cards", "description": "Получить карточки товаров",
     "input_schema": {"type": "object", "properties": {"limit": {"type": "integer"}}, "required": []}},
    {"name": "get_card_by_nm", "description": "Получить карточку товара по артикулу WB",
     "input_schema": {"type": "object", "properties": {
         "nm_id": {"type": "integer"}}, "required": ["nm_id"]}},

    # Аналитика
    {"name": "get_sales_report", "description": "Отчёт о продажах за период",
     "input_schema": {"type": "object", "properties": {
         "date_from": {"type": "string"}, "date_to": {"type": "string"}},
         "required": ["date_from", "date_to"]}},
]

SYSTEM_PROMPT = """Ты — умный агент управления магазином на Wildberries.
Умеешь: отзывы, вопросы, цены, скидки, заказы FBS, поставки, склад, реклама, карточки, аналитика.

Правила:
- Отвечай на русском языке, кратко и по делу
- При изменении цен — сначала получи список товаров, посчитай новые цены, потом изменяй
- При ответах на отзывы — вежливо, только о товаре, без рекламы и ссылок (правила WB)
- После выполнения задачи — кратко отчитайся что сделано
- Если нужны уточнения — спроси
- Если просят поднять/снизить цены на % — получи все товары, посчитай и измени все сразу одним вызовом update_prices
- Даты всегда в формате YYYY-MM-DD"""


def run_tool(wb: WBClient, name: str, inp: dict) -> str:
    try:
        # Отзывы
        if name == "get_reviews":
            r = wb.get_reviews(inp.get("is_answered", False), inp.get("take", 20))
            items = r.get("data", {}).get("feedbacks", [])
            if not items: return "Отзывов нет."
            out = []
            for x in items:
                photos = x.get("photoLinks", []) or []
                photo_info = f" [фото: {len(photos)} шт]" if photos else ""
                out.append(f"ID:{x.get('id')} | {'⭐'*x.get('productValuation',0)} | {x.get('subjectName','—')}{photo_info}\n{x.get('userName','Покупатель')}: {x.get('text','(без текста)')}")
            return "\n\n".join(out)

        elif name == "reply_to_review":
            wb.reply_review(inp["review_id"], inp["text"])
            return f"✅ Ответ опубликован на отзыв {inp['review_id']}"

        elif name == "get_review_count":
            r = wb.get_review_count()
            return json.dumps(r, ensure_ascii=False)

        # Вопросы
        elif name == "get_questions":
            r = wb.get_questions(inp.get("is_answered", False), inp.get("take", 20))
            items = r.get("data", {}).get("questions", [])
            if not items: return "Вопросов нет."
            out = []
            for x in items:
                out.append(f"ID:{x.get('id')} | {x.get('productName','—')}\n{x.get('userName','Покупатель')}: {x.get('text','')}")
            return "\n\n".join(out)

        elif name == "reply_to_question":
            wb.reply_question(inp["question_id"], inp["text"])
            return f"✅ Ответ опубликован на вопрос {inp['question_id']}"

        # Цены
        elif name == "get_goods":
            r = wb.get_goods(limit=inp.get("limit", 100))
            goods = r.get("data", {}).get("listGoods", [])
            if not goods: return "Товары не найдены."
            out = []
            for g in goods:
                sizes = g.get("sizes", [{}])
                price = sizes[0].get("price", 0) if sizes else 0
                disc = sizes[0].get("discountedPrice", 0) if sizes else 0
                out.append(f"nmID:{g.get('nmID')} | {g.get('vendorCode','—')} | {price}₽ → {disc}₽ (скидка {g.get('discount',0)}%)")
            return "\n".join(out)

        elif name == "update_prices":
            r = wb.update_prices(inp["items"])
            return f"✅ Цены обновлены для {len(inp['items'])} товаров. Ответ: {json.dumps(r, ensure_ascii=False)[:200]}"

        # Заказы
        elif name == "get_new_orders":
            r = wb.get_new_orders()
            orders = r.get("orders", [])
            if not orders: return "Новых заказов нет."
            return f"Новых заказов: {len(orders)}\n" + "\n".join(
                f"- #{o.get('id')} | {o.get('article','—')} | {o.get('price',0)/100:.0f}₽ | склад:{o.get('warehouseId','—')}" for o in orders[:15])

        elif name == "get_orders":
            r = wb.get_orders(inp["date_from"])
            orders = r.get("orders", [])
            return f"Заказов с {inp['date_from']}: {len(orders)}\n" + "\n".join(
                f"- #{o.get('id')} | {o.get('article','—')} | статус:{o.get('wbStatus','—')}" for o in orders[:15])

        elif name == "cancel_order":
            wb.cancel_order(inp["order_id"])
            return f"✅ Заказ {inp['order_id']} отменён"

        # Поставки
        elif name == "get_supplies":
            r = wb.get_supplies(inp.get("limit", 10))
            items = r.get("supplies", [])
            if not items: return "Поставок нет."
            return "\n".join(f"- {s.get('id')} | {s.get('name','—')} | {s.get('status','—')} | {s.get('createdAt','')[:10]}" for s in items)

        elif name == "create_supply":
            r = wb.create_supply(inp["name"])
            return f"✅ Создана поставка: ID {r.get('id')} — {inp['name']}"

        elif name == "add_order_to_supply":
            wb.add_order_to_supply(inp["supply_id"], inp["order_id"])
            return f"✅ Заказ {inp['order_id']} добавлен в поставку {inp['supply_id']}"

        elif name == "close_supply":
            wb.close_supply(inp["supply_id"])
            return f"✅ Поставка {inp['supply_id']} закрыта"

        elif name == "get_supply_orders":
            r = wb.get_supply_orders(inp["supply_id"])
            orders = r.get("orders", [])
            return f"В поставке {inp['supply_id']}: {len(orders)} заказов\n" + "\n".join(
                f"- #{o.get('id')} | {o.get('article','—')}" for o in orders[:20])

        # Склад
        elif name == "get_warehouses":
            r = wb.get_warehouses()
            wh = r if isinstance(r, list) else r.get("warehouses", [])
            return "\n".join(f"- ID:{w.get('id')} | {w.get('name','—')} | {w.get('city','—')}" for w in wh)

        elif name == "update_stocks":
            wb.update_stocks(inp["warehouse_id"], inp["stocks"])
            return f"✅ Остатки обновлены на складе {inp['warehouse_id']} для {len(inp['stocks'])} SKU"

        # Реклама
        elif name == "get_adverts":
            r = wb.get_adverts(inp.get("status"))
            adverts = r if isinstance(r, list) else []
            if not adverts: return "Рекламных кампаний нет."
            status_map = {4:"готова", 7:"завершена", 8:"отказ", 9:"активна", 11:"пауза"}
            return "\n".join(f"- ID:{a.get('advertId')} | {a.get('name','—')} | {status_map.get(a.get('status',0), a.get('status','—'))} | бюджет:{a.get('budget',0)}₽" for a in adverts[:20])

        elif name == "get_advert_stat":
            r = wb.get_advert_stat(inp["advert_id"], inp["date_from"], inp["date_to"])
            return json.dumps(r, ensure_ascii=False)[:1000]

        elif name == "start_advert":
            wb.start_advert(inp["advert_id"])
            return f"✅ Кампания {inp['advert_id']} запущена"

        elif name == "pause_advert":
            wb.pause_advert(inp["advert_id"])
            return f"✅ Кампания {inp['advert_id']} поставлена на паузу"

        elif name == "set_advert_budget":
            wb.set_advert_budget(inp["advert_id"], inp["amount"])
            return f"✅ Бюджет кампании {inp['advert_id']} пополнен на {inp['amount']}₽"

        elif name == "set_advert_cpm":
            wb.set_advert_cpm(inp["advert_id"], inp["cpm"], inp["param"])
            return f"✅ Ставка кампании {inp['advert_id']} изменена на {inp['cpm']} CPM"

        # Карточки
        elif name == "get_cards":
            r = wb.get_cards(limit=inp.get("limit", 10))
            cards = r.get("cards", [])
            if not cards: return "Карточек нет."
            return "\n".join(f"- nmID:{c.get('nmID')} | {c.get('vendorCode','—')} | {c.get('subjectName','—')} | фото:{len(c.get('photos',[]))}" for c in cards)

        elif name == "get_card_by_nm":
            r = wb.get_card_by_nm(inp["nm_id"])
            cards = r.get("cards", [])
            if not cards: return f"Карточка {inp['nm_id']} не найдена."
            return json.dumps(cards[0], ensure_ascii=False)[:2000]

        # Аналитика
        elif name == "get_sales_report":
            r = wb.get_sales_report(inp["date_from"], inp["date_to"])
            rows = r if isinstance(r, list) else []
            if not rows: return "Данных нет."
            total_sum = sum(x.get("retail_price_withdisc_rub", 0) for x in rows)
            total_qty = sum(x.get("quantity", 0) for x in rows)
            return f"Продажи {inp['date_from']} — {inp['date_to']}:\nЗаказов: {total_qty} | Сумма: {total_sum:,.0f}₽\n(строк в отчёте: {len(rows)})"

        return f"Инструмент {name} не найден"
    except httpx.HTTPStatusError as e:
        return f"Ошибка WB API {e.response.status_code}: {e.response.text[:300]}"
    except Exception as e:
        return f"Ошибка: {str(e)}"


# ─── Эндпоинты ────────────────────────────────────────────────────────────────

@app.post("/agent")
async def agent_endpoint(req: AgentRequest):
    try:
        claude = anthropic.Anthropic(api_key=req.anthropic_key)
        wb = WBClient(req.wb_token)
        messages = [{"role": "user", "content": req.task}]
        steps = []

        for _ in range(15):
            response = claude.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                final = "".join(b.text for b in response.content if hasattr(b, "text"))
                return {"success": True, "result": final, "steps": steps}

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        steps.append(f"🔧 {block.name}")
                        result = run_tool(wb, block.name, block.input)
                        steps.append(f"✅ {result[:80]}...")
                        tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
                messages.append({"role": "user", "content": tool_results})

        return {"success": False, "result": "Превышен лимит шагов", "steps": steps}
    except anthropic.AuthenticationError:
        raise HTTPException(400, "Неверный Anthropic API ключ")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/reviews")
async def get_reviews(wb_token: str, is_answered: bool = False, take: int = 20):
    try:
        wb = WBClient(wb_token)
        return wb.get_reviews(is_answered, take)
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/reviews/generate-reply")
async def generate_reply(req: GenerateReplyRequest):
    try:
        claude = anthropic.Anthropic(api_key=req.anthropic_key)
        prompt = f"""Напиши вежливый ответ продавца на отзыв покупателя на Wildberries.
Верни ТОЛЬКО текст ответа, без кавычек и пояснений.

Оценка: {req.stars} из 5 звёзд
Товар: {req.product}
Покупатель: {req.user_name}
Текст отзыва: {req.text or '(без текста)'}

Правила WB: только о товаре, без рекламы, без ссылок, вежливо, до 1000 символов."""
        r = claude.messages.create(
            model="claude-sonnet-4-6", max_tokens=500,
            messages=[{"role": "user", "content": prompt}])
        return {"success": True, "reply": r.content[0].text}
    except anthropic.AuthenticationError:
        raise HTTPException(400, "Неверный Anthropic API ключ")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/reviews/generate-all")
async def generate_all_replies(req: GenerateAllRequest):
    try:
        claude = anthropic.Anthropic(api_key=req.anthropic_key)
        results = []
        for rev in req.reviews:
            prompt = f"""Напиши вежливый ответ продавца на отзыв на Wildberries.
Верни ТОЛЬКО текст ответа, без кавычек.
Оценка: {rev.get('stars',5)} из 5 | Товар: {rev.get('product','—')} | Покупатель: {rev.get('userName','Покупатель')}
Отзыв: {rev.get('text','(без текста)')}
Правила: только о товаре, без рекламы, вежливо, до 1000 символов."""
            r = claude.messages.create(
                model="claude-sonnet-4-6", max_tokens=400,
                messages=[{"role": "user", "content": prompt}])
            results.append({"review_id": rev.get("id"), "reply": r.content[0].text})
        return {"success": True, "results": results}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/reviews/publish")
async def publish_reply(req: PublishReplyRequest):
    try:
        wb = WBClient(req.wb_token)
        wb.reply_review(req.review_id, req.text)
        return {"success": True}
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)
    except Exception as e:
        raise HTTPException(500, str(e))


app.mount("/", StaticFiles(directory="static", html=True), name="static")
