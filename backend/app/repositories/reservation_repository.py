"""
Reservation (Order) Repository.

Handles database operations for Reservation entities.
"""

from uuid import UUID

import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.datetime_utils import utc_now
from app.models.reservation import Reservation, ReservationStatus
from app.repositories.base_repository import BaseRepository


class ReservationRepository(BaseRepository[Reservation]):
    def __init__(self, session: AsyncSession):
        super().__init__(Reservation, session)

    async def get_by_tenant(
        self,
        tenant_id: UUID,
        limit: int = 100,
        offset: int = 0,
        status: ReservationStatus | None = None,
    ) -> list[Reservation]:
        query = (
            select(Reservation)
            .where(Reservation.tenant_id == tenant_id)
            .order_by(Reservation.created_at.desc())
            .limit(limit)
            .offset(offset)
        )

        if status:
            query = query.where(Reservation.status == status)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_by_conversation(self, conversation_id: UUID) -> list[Reservation]:
        query = (
            select(Reservation)
            .where(Reservation.conversation_id == conversation_id)
            .order_by(Reservation.created_at.desc())
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_latest_pending_by_conversation(
        self,
        conversation_id: UUID,
    ) -> Reservation | None:
        query = (
            select(Reservation)
            .where(
                Reservation.conversation_id == conversation_id,
                Reservation.status == ReservationStatus.PENDING,
            )
            .order_by(Reservation.created_at.desc())
            .limit(1)
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def find_by_idempotency_key(self, idempotency_key: str) -> Reservation | None:
        if not idempotency_key:
            return None
        query = select(Reservation).where(
            Reservation.idempotency_key == idempotency_key
        ).limit(1)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_by_order_number(self, tenant_id: UUID, order_number: str) -> Reservation | None:
        return await self.find_one_by(tenant_id=tenant_id, order_number=order_number)

    async def update_status(
        self, id: UUID, new_status: ReservationStatus, notes: str | None = None
    ) -> Reservation | None:
        update_data = {"status": new_status}

        if notes:
            update_data["notes"] = notes

        return await self.update(id, **update_data)

    async def confirm(self, id: UUID) -> Reservation | None:
        return await self.update(id, status=ReservationStatus.CONFIRMED, confirmed_at=utc_now())

    async def complete(self, id: UUID) -> Reservation | None:
        return await self.update(id, status=ReservationStatus.COMPLETED, completed_at=utc_now())

    async def start_progress(self, id: UUID) -> Reservation | None:
        return await self.update(
            id,
            status=ReservationStatus.IN_PROGRESS,
        )

    async def cancel(self, id: UUID, reason: str | None = None) -> Reservation | None:
        update_data = {"status": ReservationStatus.CANCELLED, "cancelled_at": utc_now()}

        if reason:
            update_data["notes"] = f"Cancelled: {reason}"

        return await self.update(id, **update_data)

    async def get_pending_reservations(self, tenant_id: UUID, limit: int = 50) -> list[Reservation]:
        return await self.get_by_tenant(tenant_id, limit=limit, status=ReservationStatus.PENDING)

    async def count_by_tenant(
        self, tenant_id: UUID, status: ReservationStatus | None = None
    ) -> int:
        filters = {"tenant_id": tenant_id}
        if status:
            filters["status"] = status
        return await self.count(**filters)

    async def generate_order_number(self, tenant_id: UUID) -> str:
        from datetime import date
        result = await self.session.execute(sa.text("SELECT nextval('order_number_seq')"))
        nextval = result.scalar_one()
        today = date.today().strftime("%Y%m%d")
        return f"IBR-{today}-{nextval:03d}"
