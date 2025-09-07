"""Configuration management for CFB Edge platform."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # API Keys
    cfbd_api_key: str = Field(default="test_key", description="College Football Data API key")
    odds_api_key: str = Field(default="test_key", description="The Odds API key")
    weather_api_key: Optional[str] = Field(None, description="Weather API key (optional)")
    
    # Data directories
    data_dir: Path = Field(default=Path("./data"), description="Base data directory")
    artifacts_dir: Path = Field(default=Path("./artifacts"), description="Model artifacts directory")
    
    # Model configuration
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    test_size: float = Field(default=0.2, description="Test set size for train/test split")
    calibration_method: str = Field(default="isotonic", description="Calibration method")
    
    # Feature engineering
    rolling_windows: list[int] = Field(default=[3, 5, 10], description="Rolling window sizes")
    
    # Backtesting
    transaction_cost: float = Field(default=0.0, description="Transaction cost per bet")
    max_kelly_fraction: float = Field(default=0.25, description="Maximum Kelly fraction")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "bronze").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "silver").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "gold").mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
