"""
Use this code to gain access to all bookmarks in Chrome to store them in your database. 
It can be used as a tool for your LLM agents, providing them access to verified web resources.

To run this code, you need to add the missing custom parts described below.
Those parts are marked by the '#CUSTOM' tag in the code: The changes made include:
- install libraries from requirements.txt,
- download Ollama & llama3.2 (or any other model providing chat interface),
- provide DB connector to your database,
- create table:
    -   folder_id: varchar(50)
    -   folder_name: varchar(100)
    -   url_guid: varchar(100)
    -   url_id: varchar(50)
    -   url_tab_name: varchar(1000)
    -   url: LONGTEXT NOT NULL
    -   description: LONGTEXT NULL
    -   status: int(11)
- adjust settings (Settings class) or replace it with .env file.
"""

import re
import httpx
import asyncio
import logging
from uuid import UUID
import chrome_bookmarks
from bs4 import BeautifulSoup
from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Any, Generator, List, Tuple
from pydantic import BaseModel, field_validator
from ollama import AsyncClient as OllamaAsyncClient

logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore.http11").setLevel(logging.ERROR)
logging.getLogger("httpcore.connection").setLevel(logging.ERROR)

#CUSTOM
# replace with your DB connector here or just fill missing parts
class Database:
    def __init__(self, connection_settings: dict, isolation_level: str = 'READ_COMMITTED') -> None: ...
      
    async def cleanup(self): ...
    async def rollback(self) -> bool: ...
    async def select(self, query: str) -> List[tuple]: ...
    async def bulk_insert(
        self, query: str, params: list[tuple], commit: bool = True
    ) -> List[int | str]: ...

#CUSTOM
# provide custom settings or replace with .env & load_dotenv()
class Settings(BaseSettings):
    # This is for demonstration purposes, in real-world scenario,
    # you'd load settings from a configuration file
    DB_TABLE: str = ""  # add table name
    HEADERS: dict = {
        "User-Agent": """Mozilla/5.0 (Windows NT 10.0; Win64; x64) 
        AppleWebKit/537.36 (KHTML, like Gecko) 
        Chrome/91.0.4472.124 Safari/537.36"""
    }
    DB_CONNECTION_STRING: dict = {
        "host": "localhost",
        "port": 3306,
        "user": "root",
        "password": "",  # add password
        "database": "",  # add database name
    }


class BookmarkUrl(BaseModel):
    guid: UUID
    id: str
    name: str
    url: str

    @field_validator("name", "url")
    @lru_cache(maxsize=200)
    def replace_double_quotes(cls, value: str) -> str:
        if isinstance(value, str):
            return value.replace('"', "'")
        return value


class BookmarkFolder(BaseModel):
    id: int
    name: str
    urls: List[BookmarkUrl]

    def __hash__(self):
        return hash((self.id, self.name))

    def __eq__(self, other):
        if not isinstance(other, BookmarkFolder):
            return NotImplemented
        return self.id == other.id and self.name == other.name


class Model:
    
    SPECIAL_CHARS: List[str] = [
        ",",".","'",'"',
        "!","#","$","%",
        "&","*","+","-",
        "/","=","?","^",
        "_","`","{","|",
        "}","~"
    ]

    SYSTEM: str = """You are a content writer specialist. 
    You deliver short, meaningful summaries without additional comments. 
    Do not use any kind of introduction."""

    def __init__(
        self, model_name: str = "llama3.2", system_message: str = None
    ) -> None:
        self.model_name = model_name
        self.model = OllamaAsyncClient()
        self.system_message = system_message if system_message else self.SYSTEM

    @lru_cache(maxsize=200)
    def remove_special_chars(self, text: str) -> str:
        for char in self.SPECIAL_CHARS:
            text = text.replace(char, "")
        return text

    @lru_cache(maxsize=1000)
    async def model_chat(self, prompt, system_message: str = None):
        answer = await self.model.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_message if system_message else self.SYSTEM,
                },
                {"role": "user", "content": prompt},
            ],
        )
        return answer

    @lru_cache(maxsize=1000)
    async def resource_description(
        self, context_data: str = None, prompt: str = None
    ) -> str:
        if not prompt:
            if not context_data:
                raise ValueError("Either prompt or context_data must be provided.")
            prompt = f"Craft a brief overview of summary of: {context_data}"
        description = await self.model_chat(prompt)
        return self.remove_special_chars(description.message.content)


class BookmarkRetriever:
    """Handle bookmark retrieval from Chrome"""

    @lru_cache(maxsize=2000)
    def retrieve_bookmarks(self) -> Generator:
        bookmarks = set()
        for folder in chrome_bookmarks.folders:
            bookmarks.add(
                BookmarkFolder(
                    id=folder.id,
                    name=folder.name,
                    urls=[BookmarkUrl(**url) for url in folder.urls],
                )
            )

        for bookmark in bookmarks:
            if bookmark.urls:
                yield bookmark.model_dump()


class WebsiteDescriptionFetcher:
    """Handle website content fetching and parsing"""

    REGEX = re.compile(r"https?://([^/]+)")
    HEADERS = Settings().HEADERS

    @lru_cache(maxsize=1000)
    def __extract_domain(self, url: str) -> str:
        match = self.REGEX.search(url)
        return f"https://{match.group(1)}" if match else ""

    @lru_cache(maxsize=2000)
    async def get_website_description(
        self, url: str, headers: dict = {}, error_info: str = "Page is not accessible"
    ) -> Tuple[str, str, int]:
        try:
            description = None
            if not headers:
                headers = self.HEADERS

            async with httpx.AsyncClient(
                follow_redirects=True, max_redirects=5, verify=False
            ) as client:
                response = await client.get(url, timeout=5, headers=headers)

            status = int(response.status_code)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                title = soup.title.string if soup.title else "No title found"
                meta_desc = soup.find("meta", attrs={"name": "description"})
                if meta_desc:
                    description = meta_desc.get("content")
                if not description:
                    main_domain = self.__extract_domain(url)
                    if main_domain and main_domain != url:
                        return await self.get_website_description(main_domain)
            else:
                title = error_info
                description = error_info
            return title, description or "", status
        except Exception:
            return error_info, error_info, 420


class BookmarksOrganizer:
    """Handle database operation"""

    def __init__(
        self,
        model: Any,
        database: Any,
        settings: BaseSettings,
        bookmarks_retriever: Any,
        website_retriever: Any,
        logger: logging.Logger = None,
    ) -> None:
        self.model = model
        self.database = database
        self.website_retriever = website_retriever
        self.bookmarks_retriever = bookmarks_retriever
        self.logger = logger if logger else self.null_logger()
        self.settings = settings

    def null_logger():
        null_logger = logging.getLogger("null_logger")
        null_logger.addHandler(logging.NullHandler())
        return null_logger

    async def insert_bookmarks(self) -> None:
        all_urls = set()
        params: List[Tuple[str, str, str, str, str, str, str]] = []
        bookmarks_in_db = await self.database.select(
            f"SELECT DISTINCT(url) FROM {self.settings.DB_TABLE}"
        )
        self.logger.debug(f"Fetched {len(bookmarks_in_db)} bookmark(s) from DB")

        all_urls.update({row[0] for row in bookmarks_in_db})
        query = f"""INSERT INTO {self.settings.DB_TABLE} (folder_id, folder_name, url_guid, url_id, url_tab_name, url, description, status) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""

        self.logger.debug("Processing bookmarks from Chrome")
        for item in self.bookmarks_retriever.retrieve_bookmarks():
            for url in item.get("urls", []):
                if url.get("url") not in all_urls:
                    title, content, status = (
                        await self.website_retriever.get_website_description(
                            url.get("url")
                        )
                    )
                    params.append(
                        (
                            str(item.get("id")),
                            str(item.get("name")),
                            str(url.get("guid")),
                            str(url.get("id")),
                            url.get("name", None),
                            url.get("url"),
                            await self.model.resource_description(
                                context_data=f"{title} {content}"
                            ),
                            int(status),
                        )
                    )
                    all_urls.add(url.get("url"))
        if params:
            self.logger.debug("Inserting bookmarks")
            try:
                await self.database.bulk_insert(query, params)
                self.logger.info("Bookmarks inserted successfully")
            except Exception as e:
                await self.database.rollback()
                self.logger.error(f"Insert rollback. Failed to insert bookmarks: {e}")
        else:
            self.logger.debug("No new bookmarks to insert")


def simple_logger(name: str = "bookmarks") -> logging.Logger:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger = logging.Logger(name=name, level=logging.DEBUG)
    logger.addHandler(console_handler)
    return logger


async def main():

    # settings
    settings = Settings()

    #CUSTOM
    # provide DB connector to run the code
    database = Database(settings=...)

    #CUSTOM
    # replace with custom logger or skip
    logger = simple_logger()

    # run code
    bookmarks = BookmarksOrganizer(
        logger=logger,
        model=Model(),
        database=database,
        settings=settings,
        bookmarks_retriever=BookmarkRetriever(),
        website_retriever=WebsiteDescriptionFetcher(),
    )

    try:
        await bookmarks.insert_bookmarks()
    except Exception as e:
        logger.error(f"Failed to insert bookmarks: {e}")
    finally:
        await database.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
