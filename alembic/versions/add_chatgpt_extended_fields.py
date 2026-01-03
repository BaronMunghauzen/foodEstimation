"""add chatgpt extended fields

Revision ID: 991fed93e8a5
Revises: 7fa53baaaf9c
Create Date: 2026-01-02 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '991fed93e8a5'
down_revision: Union[str, None] = '7fa53baaaf9c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Добавляем новые JSON поля для расширенной информации от ChatGPT
    op.add_column('food_requests', sa.Column('chatgpt_ingredients', sa.JSON(), nullable=True))
    op.add_column('food_requests', sa.Column('chatgpt_recommendations', sa.JSON(), nullable=True))
    op.add_column('food_requests', sa.Column('chatgpt_micronutrients', sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    # Удаляем добавленные поля
    op.drop_column('food_requests', 'chatgpt_micronutrients')
    op.drop_column('food_requests', 'chatgpt_recommendations')
    op.drop_column('food_requests', 'chatgpt_ingredients')

