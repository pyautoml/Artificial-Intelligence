import os
import asyncio
from dotenv import load_dotenv

# custom imports
from database.database_maria_db import MariaDB
from utilities.custom_logger import setup_logger
from functions.chrome_bookmarks import (
    BookmarkModel, 
    BookmarkRetriever, 
    BookmarksOrganizer, 
    WebsiteDescriptionFetcher
)

load_dotenv()

async def main():

    # providing default script logger
    logger = setup_logger(
        name=os.environ.get("bookmarks_logger_name"), log_level="DEBUG"
    )

    # settings
    db_settings: dict = {
        "host": os.environ.get("db_host"),
        "port": int(os.environ.get("db_port")),
        "user": os.environ.get("db_user"),
        "password": os.environ.get("db_password"),
        "database": os.environ.get("db_database"),
    }

    # provide DB connector to run the code
    database = MariaDB(connection_settings=db_settings, logger=logger)
    await database._connect(isolation_level="READ_COMMITTED")

    # run code
    bookmarks = BookmarksOrganizer(
        model=BookmarkModel(),
        database=database,
        bookmarks_retriever=BookmarkRetriever(),
        website_retriever=WebsiteDescriptionFetcher(),
    )

    try:
        await bookmarks._insert_bookmarks()
    except Exception as e:
        logger.error(f"Failed to insert bookmarks: {e}")
    finally:
        await database._disconnect()


if __name__ == "__main__":
    asyncio.run(main())
