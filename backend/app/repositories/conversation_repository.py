"""
Conversation Repository.

Handles database operations for Conversation and Message entities.
"""

from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.datetime_utils import utc_now
from app.core.pagination import decode_cursor, encode_cursor
from app.models.conversation import Conversation, ConversationState, Message, MessageRole
from app.repositories.base_repository import BaseRepository


class ConversationRepository(BaseRepository[Conversation]):
    def __init__(self, session: AsyncSession):
        super().__init__(Conversation, session)

    async def get_by_tenant_and_user(
        self, tenant_id: UUID, social_user_id: str
    ) -> Conversation | None:
        query = (
            select(Conversation)
            .where(
                Conversation.tenant_id == tenant_id, Conversation.social_user_id == social_user_id
            )
            .order_by(Conversation.last_message_at.desc())
        )

        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_or_create_conversation(
        self, tenant_id: UUID, social_user_id: str, user_name: str | None = None
    ) -> Conversation:
        conversation = await self.get_by_tenant_and_user(tenant_id, social_user_id)

        if conversation:
            if conversation.state == ConversationState.COMPLETED:
                await self.update(conversation.id, state=ConversationState.BROWSING, ai_paused=False)
                conversation.state = ConversationState.BROWSING
                conversation.ai_paused = False
            if user_name and conversation.user_name in (None, "Unknown User"):
                await self.update(conversation.id, user_name=user_name)
                conversation.user_name = user_name
            return conversation

        stmt = (
            insert(Conversation)
            .values(
                tenant_id=tenant_id,
                social_user_id=social_user_id,
                user_name=user_name or "Unknown User",
                state=ConversationState.BROWSING,
                is_within_24h=True,
                ai_paused=False,
            )
            .on_conflict_do_nothing(
                constraint="uq_conversations_tenant_social_user",
            )
            .returning(Conversation.id)
        )

        result = await self.session.execute(stmt)
        conversation_id = result.scalar_one_or_none()

        if conversation_id:
            return await self.get_by_id(conversation_id)

        existing = await self.get_by_tenant_and_user(tenant_id, social_user_id)
        if not existing:
            raise RuntimeError("Failed to create or fetch conversation")
        return existing

    async def get_active_conversations(
        self, tenant_id: UUID, limit: int = 50, offset: int = 0
    ) -> list[Conversation]:
        query = (
            select(Conversation)
            .where(
                Conversation.tenant_id == tenant_id,
                Conversation.state != ConversationState.COMPLETED,
            )
            .order_by(Conversation.last_message_at.desc())
            .limit(limit)
            .offset(offset)
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_all_conversations(
        self,
        tenant_id: UUID,
        state: ConversationState | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Conversation]:
        query = (
            select(Conversation)
            .where(Conversation.tenant_id == tenant_id)
            .order_by(Conversation.last_message_at.desc())
            .limit(limit)
            .offset(offset)
        )

        if state:
            query = query.where(Conversation.state == state)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def count_by_tenant(self, tenant_id: UUID, state: ConversationState | None = None) -> int:
        filters = {"tenant_id": tenant_id}
        if state:
            filters["state"] = state
        return await self.count(**filters)

    async def get_message_count(self, conversation_id: UUID, tenant_id: UUID | None = None) -> int:
        from sqlalchemy import func as sa_func

        query = (
            select(sa_func.count())
            .select_from(Message)
            .where(Message.conversation_id == conversation_id)
        )
        if tenant_id is not None:
            query = query.where(Message.tenant_id == tenant_id)
        result = await self.session.execute(query)
        return result.scalar_one()

    async def get_last_message_preview(
        self, conversation_id: UUID, max_length: int = 100, tenant_id: UUID | None = None
    ) -> str | None:
        query = (
            select(Message.content)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(1)
        )
        if tenant_id is not None:
            query = query.where(Message.tenant_id == tenant_id)
        result = await self.session.execute(query)
        content = result.scalar_one_or_none()
        if content and len(content) > max_length:
            return content[:max_length] + "..."
        return content

    async def batch_get_stats(
        self,
        conversation_ids: list[UUID],
        preview_max_length: int = 100,
        tenant_id: UUID | None = None,
    ) -> dict[UUID, dict]:
        from sqlalchemy import func as sa_func

        if not conversation_ids:
            return {}

        count_q = (
            select(
                Message.conversation_id,
                sa_func.count(Message.id).label("cnt"),
            )
            .where(Message.conversation_id.in_(conversation_ids))
            .group_by(Message.conversation_id)
        )
        if tenant_id is not None:
            count_q = count_q.where(Message.tenant_id == tenant_id)
        count_rows = (await self.session.execute(count_q)).all()
        counts: dict[UUID, int] = {row.conversation_id: row.cnt for row in count_rows}

        preview_q = (
            select(Message.conversation_id, Message.content)
            .distinct(Message.conversation_id)
            .where(Message.conversation_id.in_(conversation_ids))
            .order_by(Message.conversation_id, Message.created_at.desc())
        )
        if tenant_id is not None:
            preview_q = preview_q.where(Message.tenant_id == tenant_id)
        preview_rows = (await self.session.execute(preview_q)).all()
        previews: dict[UUID, str | None] = {}
        for row in preview_rows:
            content = row.content
            if content and len(content) > preview_max_length:
                content = content[:preview_max_length] + "..."
            previews[row.conversation_id] = content

        return {
            cid: {
                "message_count": counts.get(cid, 0),
                "last_message_preview": previews.get(cid),
            }
            for cid in conversation_ids
        }

    async def update_state(self, id: UUID, new_state: ConversationState) -> Conversation | None:
        return await self.update(id, state=new_state)

    async def update_24h_window(self, id: UUID, is_within_24h: bool) -> Conversation | None:
        return await self.update(id, is_within_24h=is_within_24h, last_message_at=utc_now())

    async def pause_ai(self, id: UUID) -> Conversation | None:
        return await self.update(id, ai_paused=True)

    async def resume_ai(self, id: UUID) -> Conversation | None:
        return await self.update(id, ai_paused=False)

    async def add_message(
        self,
        conversation_id: UUID,
        role: MessageRole,
        content: str,
        meta_data: dict | None = None,
        meta_message_id: str | None = None,
    ) -> Message:
        conversation = await self.get_by_id(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        message = Message(
            tenant_id=conversation.tenant_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            meta_data=meta_data or {},
            meta_message_id=meta_message_id,
        )

        self.session.add(message)
        await self.session.flush()
        await self.session.refresh(message)

        await self.update(conversation_id, last_message_at=utc_now())

        return message

    async def get_conversation_messages(
        self,
        conversation_id: UUID,
        limit: int = 20,
        cursor: str | None = None,
        tenant_id: UUID | None = None,
    ) -> tuple[list[Message], str | None, bool]:
        query = select(Message).where(Message.conversation_id == conversation_id)
        if tenant_id is not None:
            query = query.where(Message.tenant_id == tenant_id)

        if cursor:
            try:
                cursor_values = decode_cursor(cursor)
                cursor_created_at, cursor_id = cursor_values[0], cursor_values[1]
            except (ValueError, IndexError) as exc:
                raise ValueError("Invalid pagination cursor") from exc

            query = query.where(
                or_(
                    Message.created_at < cursor_created_at,
                    and_(
                        Message.created_at == cursor_created_at,
                        Message.id < cursor_id,
                    ),
                )
            )

        query = query.order_by(Message.created_at.desc(), Message.id.desc()).limit(limit + 1)

        result = await self.session.execute(query)
        rows = list(result.scalars().all())

        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]

        next_cursor: str | None = None
        if has_more and rows:
            oldest = rows[-1]
            next_cursor = encode_cursor([oldest.created_at, oldest.id])

        rows.reverse()
        return rows, next_cursor, has_more

    async def get_recent_context(
        self, conversation_id: UUID, num_messages: int = 10
    ) -> list[Message]:
        query = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(num_messages)
        )
        result = await self.session.execute(query)
        messages = list(result.scalars().all())

        return list(reversed(messages))
