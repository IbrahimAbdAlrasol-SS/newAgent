"""
RAG Retrieval Node.

Retrieves relevant products using semantic search.
Triggered when intent is product_inquiry or price_check.
"""

from uuid import UUID

from loguru import logger

from ai_engine.agents.state import ConversationState
from ai_engine.analytics.metrics import (
    RAG_INFRA_FAILURE_COUNTER,
    RAG_RESULTS_COUNTER,
    RAG_RETRIEVAL_DURATION,
    observe_duration,
)
from ai_engine.embeddings.embedding_service import EmbeddingService
from ai_engine.rag.vector_store import MultiTenantVectorStore

import re


# --- Concept synonym dictionary (Arabic ↔ English) ---
# Used by `_db_keyword_search` as a poor-man's semantic search: when the user
# asks "بدي شي للوجه" we expand "وجه" → {وجه, بشرة, face, skin} and ILIKE each
# variant, so a product named "كريم بشرة" still matches.
_CONCEPT_SYNONYMS: dict[str, list[str]] = {
    # --- Beauty / Skincare ---
    "وجه": ["وجه", "بشرة", "face", "skin", "facial"],
    "بشرة": ["بشرة", "وجه", "skin", "face"],
    "كريم": ["كريم", "cream", "مرطب", "moisturizer", "lotion"],
    "مرطب": ["مرطب", "كريم", "moisturizer", "cream", "hydrating"],
    "شعر": ["شعر", "hair", "شامبو", "shampoo", "بلسم", "conditioner"],
    "شامبو": ["شامبو", "shampoo", "شعر", "hair"],
    "عطر": ["عطر", "perfume", "fragrance", "بخاخ", "كولونيا", "cologne"],
    "مكياج": ["مكياج", "makeup", "تجميل", "cosmetic", "cosmetics"],
    "احمر": ["أحمر", "احمر", "lipstick", "روج", "شفايف", "lips"],
    "شفايف": ["شفايف", "lips", "روج", "lipstick"],
    "عيون": ["عيون", "عين", "eye", "eyes", "كحل", "eyeliner", "mascara", "ماسكارا"],
    "اظافر": ["أظافر", "اظافر", "nail", "nails", "مناكير"],
    "حماية": ["حماية", "sunscreen", "sun", "spf", "شمس"],
    "شمس": ["شمس", "sun", "sunscreen", "spf", "حماية"],
    "تنظيف": ["تنظيف", "غسول", "cleanser", "wash", "soap", "صابون"],
    "غسول": ["غسول", "cleanser", "wash", "تنظيف"],
    "حب": ["حب", "حبوب", "acne", "pimples", "بثور"],
    "بثور": ["بثور", "حب", "حبوب", "acne", "pimples"],
    "تجاعيد": ["تجاعيد", "wrinkles", "anti-aging", "antiaging"],
    "تبييض": ["تبييض", "whitening", "brightening", "تفتيح"],
    "يدين": ["يدين", "يد", "hand", "hands"],
    "قدم": ["قدم", "اقدام", "أقدام", "foot", "feet"],
    "جسم": ["جسم", "body"],
    # --- Clothing / Shoes ---
    "ملابس": ["ملابس", "لباس", "clothing", "clothes", "wear"],
    "فستان": ["فستان", "فساتين", "dress"],
    "قميص": ["قميص", "shirt", "تيشيرت", "t-shirt", "tshirt"],
    "بنطلون": ["بنطلون", "بنطال", "pants", "trousers", "جينز", "jeans"],
    "حذاء": ["حذاء", "أحذية", "جزمة", "shoe", "shoes", "sneaker", "sneakers"],
    "شنطة": ["شنطة", "حقيبة", "حقائب", "bag", "handbag"],
    "ساعة": ["ساعة", "ساعات", "watch"],
    # --- Food / Drinks ---
    "اكل": ["أكل", "اكل", "طعام", "food", "meal"],
    "حلويات": ["حلويات", "حلو", "حلى", "dessert", "sweets", "candy"],
    "قهوة": ["قهوة", "coffee", "كافيه"],
    "شاي": ["شاي", "tea"],
    "عصير": ["عصير", "juice", "مشروب", "drink"],
    # --- Electronics ---
    "موبايل": ["موبايل", "جوال", "هاتف", "phone", "mobile", "smartphone"],
    "لابتوب": ["لابتوب", "laptop", "كمبيوتر", "computer", "pc"],
    "سماعات": ["سماعات", "سماعة", "headphones", "earbuds", "headset"],
    # --- Generic ---
    "هدية": ["هدية", "هدايا", "gift", "present"],
    "رخيص": ["رخيص", "ارخص", "أرخص", "cheap", "budget", "أقل سعر"],
    "غالي": ["غالي", "premium", "luxury", "فاخر"],
}

# Arabic prefixes commonly attached to nouns ("للوجه" → "وجه").
# Single-letter prefixes (و، ف) are intentionally omitted — they over-strip
# valid roots and add noise.
_AR_PREFIXES = ("للـ", "بال", "وال", "كال", "فال", "ال", "لل", "بـ", "لـ")


def _strip_ar_prefix(token: str) -> str:
    """Strip common Arabic prefixes (ال، لل، بـ، ...) for keyword matching."""
    for p in _AR_PREFIXES:
        if token.startswith(p) and len(token) > len(p) + 1:
            return token[len(p):]
    return token


def _expand_tokens(tokens: list[str]) -> list[str]:
    """Expand each token with its synonyms (Arabic ↔ English)."""
    expanded: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        base = _strip_ar_prefix(tok.lower())
        candidates = [tok, base]
        # Synonym lookup (case-insensitive)
        for key, vals in _CONCEPT_SYNONYMS.items():
            if base == key.lower() or tok.lower() == key.lower():
                candidates.extend(vals)
                break
            if base in (v.lower() for v in vals) or tok.lower() in (v.lower() for v in vals):
                candidates.extend(vals)
                break
        for c in candidates:
            cl = c.strip().lower()
            if len(cl) >= 2 and cl not in seen:
                seen.add(cl)
                expanded.append(c.strip())
    return expanded


# Arabic filler words to strip before embedding queries
_AR_FILLERS = re.compile(
    r"\b(?:يعني|هل|ممكن|لو\s*سمحت|عندكم|في|شو)\b",
    re.UNICODE,
)
# English filler phrases to strip before embedding queries
_EN_FILLERS = re.compile(
    r"\b(?:do you have|can i get|i want|please)\b",
    re.IGNORECASE,
)
# Punctuation and emoji pattern
_PUNCT_EMOJI_RE = re.compile(
    r"[^\w\s]"
    r"|[\U0001F600-\U0001F9FF\U0001FA00-\U0001FAFF\U00002702-\U000027B0]",
    re.UNICODE,
)
_MULTI_SPACE_RE = re.compile(r"\s+")


def _preprocess_query(text: str) -> str:
    """Strip filler words, punctuation, and emojis to improve embedding retrieval."""
    result = _AR_FILLERS.sub(" ", text)
    result = _EN_FILLERS.sub(" ", result)
    result = _PUNCT_EMOJI_RE.sub(" ", result)
    result = _MULTI_SPACE_RE.sub(" ", result).strip()
    return result or text


async def _db_keyword_search(
    tenant_uuid: UUID,
    query: str,
    limit: int = 8,
    include_oos: bool = False,
) -> list[dict]:
    """
    PostgreSQL keyword fallback when vector search is unavailable.

    Uses ILIKE across name / name_ar / description / description_ar / tags.
    Returns products in the same shape as RAG results so downstream nodes
    don't need to know about the fallback.
    """
    try:
        from sqlalchemy import or_, select, func
        from app.db.session import AsyncSessionLocal
        from app.models.product import Product
    except Exception as e:
        logger.warning(f"DB keyword fallback unavailable (import failed): {e}")
        return []

    raw_tokens = [t for t in re.split(r"\s+", query.strip()) if len(t) >= 2]
    if not raw_tokens:
        raw_tokens = [query.strip()] if query.strip() else []
    if not raw_tokens:
        return []

    # Expand tokens via synonym dictionary (وجه → بشرة, face, skin, ...)
    tokens = _expand_tokens(raw_tokens)
    logger.debug(f"DB keyword search: {raw_tokens} → expanded to {tokens[:15]}")

    try:
        async with AsyncSessionLocal() as session:
            conditions = []
            for tok in tokens[:15]:
                pat = f"%{tok}%"
                conditions.append(Product.name.ilike(pat))
                conditions.append(Product.name_ar.ilike(pat))
                conditions.append(Product.description.ilike(pat))
                conditions.append(Product.description_ar.ilike(pat))
                conditions.append(Product.category.ilike(pat))

            stmt = (
                select(Product)
                .where(Product.tenant_id == tenant_uuid)
                .where(Product.is_active.is_(True))
            )
            if not include_oos:
                stmt = stmt.where(Product.stock_quantity > 0)
            
            stmt = stmt.where(or_(*conditions)).limit(limit)
            rows = (await session.execute(stmt)).scalars().all()

            # If nothing matched and the catalog is small, return all active
            # products so the AI can offer what's available instead of denying.
            if not rows:
                count_stmt = (
                    select(func.count(Product.id))
                    .where(Product.tenant_id == tenant_uuid)
                    .where(Product.is_active.is_(True))
                )
                if not include_oos:
                    count_stmt = count_stmt.where(Product.stock_quantity > 0)
                    
                total = (await session.execute(count_stmt)).scalar() or 0
                if 0 < total <= 25:
                    fallback_stmt = select(Product).where(Product.tenant_id == tenant_uuid).where(Product.is_active.is_(True))
                    if not include_oos:
                        fallback_stmt = fallback_stmt.where(Product.stock_quantity > 0)
                    rows = (await session.execute(fallback_stmt.limit(limit))).scalars().all()

            results: list[dict] = []
            for p in rows:
                results.append({
                    "id": str(p.id),
                    "name": p.name or "",
                    "name_ar": p.name_ar or "",
                    "price": float(p.price) if p.price is not None else 0,
                    "description": p.description or "",
                    "description_ar": p.description_ar or "",
                    "brand": (p.attributes or {}).get("brand", "") if isinstance(p.attributes, dict) else "",
                    "similarity_score": 0.5,  # neutral score — keyword match
                    "metadata": dict(p.attributes or {}) if isinstance(p.attributes, dict) else {},
                })
            return results
    except Exception as e:
        logger.warning(f"DB keyword fallback query failed: {type(e).__name__}: {e}")
        return []

# Default similarity threshold — overridden by settings.RAG_SIMILARITY_THRESHOLD
_DEFAULT_SIMILARITY_THRESHOLD = 0.35  # cosine similarity lower bound

try:
    from app.core.config import settings as _backend_settings

    _DEFAULT_SIMILARITY_THRESHOLD = _backend_settings.RAG_SIMILARITY_THRESHOLD
except Exception:
    try:
        from backend.app.core.config import settings as _backend_settings

        _DEFAULT_SIMILARITY_THRESHOLD = _backend_settings.RAG_SIMILARITY_THRESHOLD
    except Exception:
        pass  # ai_engine can run standalone without backend settings


class RAGRetrievalNode:
    """
    Retrieve relevant products using RAG.

    Uses embeddings and vector store to find semantically similar products
    based on user query. Filters results to only include active products
    and removes low-relevance results below the similarity threshold.

    Example:
        ```python
        retrieval = RAGRetrievalNode(embedding_service, vector_store)

        state = ConversationState(
            messages=[{"role": "user", "content": "فساتين حمراء"}],
            tenant_id="store_123",
            social_user_id="user_456"
        )

        updates = await retrieval(state)
        print(len(updates["retrieved_products"]))  # Number of products found
        ```
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: MultiTenantVectorStore,
        similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
    ):
        """
        Initialize RAG retrieval node.

        Args:
            embedding_service: Service for generating embeddings
            vector_store: Vector store for similarity search
            similarity_threshold: Minimum cosine similarity to include a result
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
        logger.info(f"Initialized RAGRetrievalNode (threshold={similarity_threshold})")

    async def __call__(self, state: ConversationState) -> dict:
        """
        Retrieve products based on user query.

        Args:
            state: Current conversation state

        Returns:
            Dict with "retrieved_products" and "rag_status" keys
        """
        if not state.messages:
            logger.warning("No messages in state")
            return {"retrieved_products": [], "rag_status": "no_results"}

        last_message = state.messages[-1].content
        cleaned_query = _preprocess_query(last_message)

        # ── Level 3: Stock Query Rate Limit & Validation ──
        from ai_engine.security.stock_guard import stock_guard
        
        # 1. Rate Limit
        if state.tenant_id and state.social_user_id:
            _rate_limited = not await stock_guard.check_query_rate_limit(
                state.social_user_id, state.tenant_id
            )
            if _rate_limited:
                logger.warning(f"RAG query rate limited for {state.social_user_id}")
                return {"retrieved_products": [], "rag_status": "rate_limited"}
        
        # 2. Validation
        validation_error = stock_guard.validate_query(cleaned_query)
        if validation_error:
            logger.warning(f"RAG query validation failed: {validation_error}")
            return {"retrieved_products": [], "rag_status": "validation_error"}

        # E1: Validate tenant_id before UUID conversion
        if not state.tenant_id or not state.tenant_id.strip():
            logger.error("Missing tenant_id in state — cannot perform RAG retrieval")
            RAG_INFRA_FAILURE_COUNTER.labels(reason="missing_tenant_id").inc()
            return {"retrieved_products": [], "rag_status": "infra_error"}

        try:
            tenant_uuid = UUID(state.tenant_id)
        except (ValueError, AttributeError) as exc:
            logger.error(f"Invalid tenant_id '{state.tenant_id}': {exc}")
            RAG_INFRA_FAILURE_COUNTER.labels(reason="invalid_tenant_id").inc()
            return {"retrieved_products": [], "rag_status": "infra_error"}

        # Vector search unavailable (no embedding service or no vector store) —
        # fall back to PostgreSQL keyword search so the AI doesn't deny products
        # that actually exist in the catalog.
        if self.embedding_service is None or self.vector_store is None:
            logger.info(
                "RAG vector search unavailable — using DB keyword fallback "
                f"(embedding={self.embedding_service is not None}, vector_store={self.vector_store is not None})"
            )
            RAG_INFRA_FAILURE_COUNTER.labels(reason="embedding_unavailable").inc()
            db_products = await _db_keyword_search(tenant_uuid, cleaned_query, limit=8)
            
            from ai_engine.security.stock_guard import stock_guard
            if db_products:
                db_products = stock_guard.sanitize_results(db_products)
                
            RAG_RESULTS_COUNTER.labels(status="found" if db_products else "empty").inc()
            return {
                "retrieved_products": db_products,
                "presented_products": db_products if db_products else (state.presented_products or []),
                "rag_status": "ok" if db_products else "no_results",
            }

        try:
            # Generate query embedding with instruction prefix
            query_embedding = await self.embedding_service.embed_query(cleaned_query)

            # Search vector store — no metadata filter so stale/missing fields
            # don't silently exclude valid products. Active-only filtering is
            # handled during product sync (only active products are indexed).
            import time as _time
            _rag_start = _time.perf_counter()
            results, search_status = await self.vector_store.search_products(
                tenant_id=tenant_uuid,
                query_embedding=query_embedding,
                top_k=8,
            )
            RAG_RETRIEVAL_DURATION.observe(_time.perf_counter() - _rag_start)
            if search_status != "ok":
                RAG_INFRA_FAILURE_COUNTER.labels(reason=search_status).inc()
                logger.warning(
                    "RAG retrieval unavailable for tenant {tenant} (status={status}) — falling back to DB",
                    tenant=state.tenant_id,
                    status=search_status,
                )
                db_products = await _db_keyword_search(tenant_uuid, cleaned_query, limit=8)
                
                from ai_engine.security.stock_guard import stock_guard
                if db_products:
                    db_products = stock_guard.sanitize_results(db_products)
                    
                if db_products:
                    RAG_RESULTS_COUNTER.labels(status="found").inc()
                    return {
                        "retrieved_products": db_products,
                        "presented_products": db_products if db_products else (state.presented_products or []),
                        "rag_status": "ok",
                    }
                else:
                    new_status = "no_results" if search_status == "collection_missing" else search_status
                    RAG_RESULTS_COUNTER.labels(status="empty").inc()
                    return {"retrieved_products": [], "rag_status": new_status}

            # Use per-tenant threshold if set in state, else fall back to instance default
            threshold = state.rag_similarity_threshold or self.similarity_threshold

            # Apply similarity threshold to filter low-relevance results
            results = [r for r in results if r.get("similarity", 0.0) >= threshold]

            # Fetch live stock from DB for the retrieved vector IDs
            valid_results = []
            oos_results = []
            if results:
                try:
                    from sqlalchemy import select
                    from app.db.session import AsyncSessionLocal
                    from app.models.product import Product
                    
                    product_ids = [UUID(str(r["id"])) for r in results if r.get("id")]
                    async with AsyncSessionLocal() as session:
                        stmt = select(Product.id, Product.stock_quantity, Product.is_active).where(Product.id.in_(product_ids))
                        rows = (await session.execute(stmt)).all()
                        
                        live_stock = {str(r.id): (r.stock_quantity, r.is_active) for r in rows}
                        
                        for r in results:
                            pid = str(r.get("id"))
                            if pid in live_stock:
                                stock, is_active = live_stock[pid]
                                if is_active and stock > 0:
                                    valid_results.append(r)
                                elif is_active and stock <= 0:
                                    r["stock_quantity"] = stock
                                    oos_results.append(r)
                except Exception as e:
                    logger.warning(f"Failed to fetch live stock for RAG results: {e}")
                    valid_results = results  # fallback to vector store assuming valid

            # Return top 5 valid products after filtering
            results = valid_results[:5]

            logger.info(
                f"Retrieved {len(results)} products (above threshold={threshold}) "
                f"for query: {last_message[:50]}"
            )

            # Convert to serializable format with key validation (E6)
            products = []
            for r in results:
                if not r.get("id"):
                    logger.warning("Skipping RAG result with missing 'id' key")
                    continue
                meta = r.get("metadata", {})
                products.append(
                    {
                        "id": r["id"],
                        "name": r.get("name", ""),
                        "name_ar": meta.get("name_ar", ""),
                        "price": r.get("price", 0),
                        "description": r.get("description", ""),
                        "description_ar": meta.get("description_ar", ""),
                        "brand": meta.get("brand", "") or meta.get("الماركة", ""),
                        "skin_type": meta.get("skin_type", "") or meta.get("نوع البشرة", ""),
                        "size": meta.get("size", "") or meta.get("الحجم", ""),
                        "ingredients": meta.get("ingredients", "") or meta.get("المكونات", ""),
                        "similarity_score": r.get("similarity", 0.0),
                        "metadata": meta,
                    }
                )

            # If RAG found nothing but previous turn had products, carry them
            # forward (e.g. user says "I want the one for 200" referencing
            # products already shown).
            if not products and state.retrieved_products:
                logger.info(
                    f"RAG returned 0 results — carrying forward "
                    f"{len(state.retrieved_products)} products from previous turn"
                )
                
                # Check for OOS query logic here as well
                oos_products = []
                if not products and oos_results:
                    for r in oos_results[:3]:
                        meta = r.get("metadata", {})
                        oos_products.append({
                            "id": r["id"],
                            "name": r.get("name", ""),
                            "name_ar": meta.get("name_ar", ""),
                            "price": r.get("price", 0),
                            "similarity_score": r.get("similarity", 0.0),
                            "stock_quantity": r.get("stock_quantity", 0),
                        })

                return {
                    "retrieved_products": state.retrieved_products,
                    "presented_products": state.retrieved_products,
                    "rag_status": "ok",
                    "oos_products": oos_products,
                }

            # Fallback: if no products found for the specific query, do a broad
            # search using business type to suggest alternatives (browse catalog)
            fallback_products = []
            if not products and state.business_type:
                _BT_QUERIES = {
                    "clothing": "ملابس أزياء",
                    "shoes": "أحذية",
                    "beauty": "مستحضرات تجميل عناية",
                    "electronics": "إلكترونيات",
                    "food": "أطعمة مشروبات",
                    "general": "منتجات",
                }
                broad_query = _BT_QUERIES.get(state.business_type, "منتجات")
                try:
                    broad_embedding = await self.embedding_service.embed_query(broad_query)
                    broad_results, broad_status = await self.vector_store.search_products(
                        tenant_id=tenant_uuid,
                        query_embedding=broad_embedding,
                        top_k=5,
                    )
                    if broad_status == "ok" and broad_results:
                        # Use a lower threshold for broad browsing
                        broad_results = [r for r in broad_results if r.get("similarity", 0.0) >= 0.15]
                        for r in broad_results[:5]:
                            if not r.get("id"):
                                continue
                            meta = r.get("metadata", {})
                            fallback_products.append({
                                "id": r["id"],
                                "name": r.get("name", ""),
                                "name_ar": meta.get("name_ar", ""),
                                "price": r.get("price", 0),
                                "description": r.get("description", ""),
                                "description_ar": meta.get("description_ar", ""),
                                "brand": meta.get("brand", "") or meta.get("الماركة", ""),
                                "similarity_score": r.get("similarity", 0.0),
                                "metadata": meta,
                            })
                        if fallback_products:
                            logger.info(
                                f"RAG primary returned 0, broad fallback found {len(fallback_products)} products"
                            )
                except Exception as broad_err:
                    logger.warning(f"RAG broad fallback failed: {broad_err}")

            # Also store as presented_products so reference resolver can use them
            RAG_RESULTS_COUNTER.labels(status="found" if products else "empty").inc()

            # Last-resort: PostgreSQL keyword fallback (handles cases where the
            # vector index is empty/stale but products exist in the catalog).
            if not products and not fallback_products:
                db_products = await _db_keyword_search(tenant_uuid, cleaned_query, limit=8)
                if db_products:
                    logger.info(
                        f"Vector search empty — DB keyword fallback found {len(db_products)} products"
                    )
                    fallback_products = db_products

            # When primary search found nothing but broad fallback did, promote
            # fallback into retrieved_products so downstream nodes see them.
            effective_products = products or fallback_products
            
            from ai_engine.security.stock_guard import stock_guard
            if effective_products:
                effective_products = stock_guard.sanitize_results(effective_products)
            
            # Format OOS products to pass to state
            oos_products_out = []
            if not effective_products and oos_results:
                for r in oos_results[:3]:
                    meta = r.get("metadata", {})
                    oos_products_out.append({
                        "id": r["id"],
                        "name": r.get("name", ""),
                        "name_ar": meta.get("name_ar", ""),
                        "price": r.get("price", 0),
                        "similarity_score": r.get("similarity", 0.0),
                        "stock_quantity": r.get("stock_quantity", 0),
                    })
            elif not effective_products and not oos_results:
                # Do a quick DB check for OOS products explicitly
                db_oos_products = await _db_keyword_search(tenant_uuid, cleaned_query, limit=3, include_oos=True)
                # Ensure they actually have 0 stock (since include_oos includes ALL products, we filter locally)
                if db_oos_products:
                    try:
                        from sqlalchemy import select
                        from app.db.session import AsyncSessionLocal
                        from app.models.product import Product
                        pids = [UUID(str(p["id"])) for p in db_oos_products]
                        async with AsyncSessionLocal() as session:
                            stmt = select(Product.id, Product.stock_quantity).where(Product.id.in_(pids))
                            rows = (await session.execute(stmt)).all()
                            live_stocks = {str(r.id): r.stock_quantity for r in rows}
                            for p in db_oos_products:
                                if live_stocks.get(str(p["id"]), 1) <= 0:
                                    oos_products_out.append(p)
                    except Exception:
                        pass
                        
            result = {
                "retrieved_products": effective_products,
                "presented_products": effective_products if effective_products else (state.presented_products or []),
                "rag_status": "ok" if effective_products else "no_results",
                "oos_products": oos_products_out,
            }
            if fallback_products and not products:
                result["fallback_products"] = fallback_products
            return result

        except Exception as e:
            logger.error(f"RAG retrieval failed: {type(e).__name__}: {e}")
            RAG_INFRA_FAILURE_COUNTER.labels(reason="exception").inc()
            # Fall back to products already in state (from previous turn)
            if state.retrieved_products:
                logger.info(
                    f"Using {len(state.retrieved_products)} products from previous turn "
                    f"as RAG fallback"
                )
                return {"retrieved_products": state.retrieved_products, "rag_status": "ok"}
            return {"retrieved_products": [], "rag_status": "infra_error"}
