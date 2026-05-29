"""Repository for PlatformSettings singleton."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.datetime_utils import utc_now
from app.models.platform_settings import PlatformSettings


class PlatformSettingsRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self) -> PlatformSettings | None:
        result = await self.session.execute(
            select(PlatformSettings).where(PlatformSettings.id == 1)
        )
        return result.scalar_one_or_none()

    async def upsert(self, **kwargs) -> PlatformSettings:
        settings = await self.get()
        if settings is None:
            settings = PlatformSettings(id=1, **kwargs)
            self.session.add(settings)
        else:
            for key, value in kwargs.items():
                if value is not None:
                    setattr(settings, key, value)
            settings.updated_at = utc_now()
        await self.session.flush()
        return settings
