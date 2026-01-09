"""add meal_plans table

Revision ID: 1d085c9be343
Revises: 991fed93e8a5
Create Date: 2026-01-02 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '1d085c9be343'
down_revision: Union[str, None] = '991fed93e8a5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'meal_plans',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('meals_per_day', sa.Integer(), nullable=False),
        sa.Column('days_count', sa.Integer(), nullable=False),
        sa.Column('target_calories', sa.Float(), nullable=False),
        sa.Column('target_proteins', sa.Float(), nullable=False),
        sa.Column('target_fats', sa.Float(), nullable=False),
        sa.Column('target_carbs', sa.Float(), nullable=False),
        sa.Column('allowed_recipes', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('chatgpt_response_raw', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('client_ip', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_meal_plans_id'), 'meal_plans', ['id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_meal_plans_id'), table_name='meal_plans')
    op.drop_table('meal_plans')

