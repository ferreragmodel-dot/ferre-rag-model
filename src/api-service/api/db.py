import os
from sqlmodel import SQLModel, create_engine, Session

_CLOUD_SQL_CONNECTION_NAME = os.environ.get("CLOUD_SQL_CONNECTION_NAME")

if _CLOUD_SQL_CONNECTION_NAME:
    # Cloud Run → Cloud SQL via Python Connector (IAM, no public IP needed)
    from google.cloud.sql.connector import Connector

    _connector = Connector()

    def _get_cloud_sql_conn():
        return _connector.connect(
            _CLOUD_SQL_CONNECTION_NAME,
            "pg8000",
            user=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
            db=os.environ["POSTGRES_DB"],
        )

    engine = create_engine("postgresql+pg8000://", creator=_get_cloud_sql_conn)
else:
    # Local dev → plain TCP to local/Docker PostgreSQL
    engine = create_engine(
        f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:"
        f"{os.environ['POSTGRES_PASSWORD']}@{os.environ['POSTGRES_HOST']}:"
        f"{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
    )


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
