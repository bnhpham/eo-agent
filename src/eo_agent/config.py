"""
Environment-based configuration.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: SecretStr | None = None
    openai_model: str | None = None

    cdse_client_id: str | None = None
    cdse_client_secret: SecretStr | None = None

    eo_data_dir: Path = PROJECT_ROOT / "data"
    upstream_registry_dir: Path = PROJECT_ROOT / "third_party" / "mcp-tool-registry"

    torch_device: str = "auto"
    prithvi_half_precision: bool = False
    max_aoi_km: float = Field(default=20.0, gt=0, le=50)
    mcp_timeout_s: float = Field(default=900.0, gt=0)
    max_pairs_per_answer: int = Field(default=10, ge=1, le=10)
    ndvi_threshold: float = Field(default=0.5, gt=0, lt=1)
    nbr_burned_dnbr: float = Field(default=0.1, gt=0, lt=0.27, description="dNBR from which a pixel counts as burned (USGS: 0.1 low severity, 0.27 moderate-low)")
    awei_threshold: float = Field(default=-0.15, ge=-2, le=2)
    awei_variant: Literal["nsh", "sh"] = "nsh"
    ndsi_threshold: float = Field(default=0.4, gt=0, lt=1)
    ndsi_min_green: float = Field(default=0.3, ge=0, lt=1)
    bsi_threshold: float = Field(default=0.0, ge=-1, le=1)
    bsi_max_ndvi: float = Field(default=0.2, ge=-1, le=1)
    gdacs_max_events: int = Field(default=50, gt=0, le=200)

    @property
    def data_dir(self) -> Path:
        return self._resolve(self.eo_data_dir)

    @property
    def registry_dir(self) -> Path:
        return self._resolve(self.upstream_registry_dir)

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def has_cdse_credentials(self) -> bool:
        return bool(self.cdse_client_id and self.cdse_client_secret and self.cdse_client_secret.get_secret_value())

    @staticmethod
    def _resolve(path: Path) -> Path:
        return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
