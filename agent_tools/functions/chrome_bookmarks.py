import os
import httpx
import logging
import inspect
from uuid import UUID
import chrome_bookmarks
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from functools import lru_cache
from pydantic import BaseModel, field_validator
from ollama import AsyncClient as OllamaAsyncClient
from typing import Any, Generator, List, Tuple, Optional

# custom imports
from utilities.custom_logger import get_logger
from database.database_abstract import AbstractDatabase
from utilities.custom_exceptions import ConwayException
from utilities.general_utils import remove_special_chars, get_random_headers, extract_domain

# switching off default logging per service
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore.http11").setLevel(logging.ERROR)
logging.getLogger("httpcore.connection").setLevel(logging.ERROR)

# loading environment variables
load_dotenv(interpolate=False)
if not os.environ.get("bookmarks_logger_name"):
    os.environ["bookmarks_logger_name"] = "bookmarks"


class BookmarkUrl(BaseModel):
    """
    Represents a bookmark URL with associated metadata.

    :param guid: (UUID) The unique identifier for the bookmark.
    :param id: (str) The ID of the bookmark.
    :param name: (str) The name of the bookmark.
    :param url: (str) The URL of the bookmark.
    """

    guid: UUID
    id: str
    name: str
    url: str

    @field_validator("name", "url")
    @lru_cache(maxsize=200)
    def replace_double_quotes(cls, value: str) -> str:
        """
        Replaces double quotes with single quotes in the given string.
        Uses lru_cache with maxsize 200 to cache the changes names and urls.

        :param value: (str) The string in which to replace double quotes.
        :returns str: The string with double quotes replaced by single quotes.
        """
        if isinstance(value, str):
            return value.replace('"', "'")
        return value


class BookmarkFolder(BaseModel):
    """
    Represents a folder containing multiple bookmarks.

    :param id: (int) The ID of the bookmark folder.
    :param name: (str) The name of the bookmark folder.
    :param urls: (List[BookmarkUrl]) A list of BookmarkUrl
     objects contained in the folder.
    """

    id: int
    name: str
    urls: List[BookmarkUrl]

    def __hash__(self):
        """
        Returns the hash value of the bookmark folder based on its ID and name.
        :returns int: The hash value of the bookmark folder.
        """
        return hash((self.id, self.name))

    def __eq__(self, other):
        """
        Checks if this bookmark folder is equal to another bookmark folder
        based on their IDs and names.

        :param other: (BookmarkFolder) The other bookmark folder to compare with.
        :returns bool: True if the bookmark folders are equal, False otherwise.
        """
        if not isinstance(other, BookmarkFolder):
            return NotImplemented
        return self.id == other.id and self.name == other.name


class BookmarkModel:
    """LLMs operations"""

    BASIC_PROMPT = """You are a content writer specialist. 
    You deliver short, meaningful summaries without 
    additional comments. Do not use any kind of introduction."""
    SYSTEM_MESSAGE: str = os.environ.get("bookmarks_prompt") or BASIC_PROMPT

    def __init__(
        self,
        model_name: Optional[str] = os.environ.get("bookmarks_model_name")
        or "llama3.2",
        system_message: Optional[str] = SYSTEM_MESSAGE,
        logger: Optional[logging.Logger] = get_logger(
            os.environ.get("bookmarks_logger_name")
        ),
    ) -> None:
        """
        Initializes the model with the specified parameters.

        :param model_name: (Optional[str]) The name of the model to be used.
         Defaults to the value of the environment variable 'bookmarks_model_name' or 'llama3.2'.
        :param system_message: (Optional[str]) The system message to be used. Defaults to SYSTEM_MESSAGE.
        :param logger: (Optional[logging.Logger]) The logger instance to be used. Defaults to a logger
         with the name specified in the environment variable 'bookmarks_logger_name'.
        """
        self.logger = logger
        self.model_name = model_name
        self.model = OllamaAsyncClient()
        self.system_message = system_message

    @lru_cache(maxsize=1000)
    async def __model_chat(self, prompt, system_message: str = None) -> str:
        """
        (async) Executes a chat request to the language model and returns the response content.
        Uses lru_cache with maxsize 1000 to cache the model's output.

        :param prompt: (str) The user prompt to be sent to the language model.
        :param system_message: (str) The system message to be sent to the language model. Defaults to None.
        :returns str: The content of the response from the language model.
        :raises ConwayException: If there is an error during the chat request.
        """
        try:
            answer = await self.model.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            system_message if system_message else self.SYSTEM_MESSAGE
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return answer.message.content
        except Exception as e:
            self.logger.error(f"Error in LLM chat with model '{self.model_name}': {e}")
            raise ConwayException(
                f"Error in LLM chat with model: {e}",
                context={
                    "answer": f"{answer}",
                    "prompt": f"{prompt}",
                    "model": f"{self.model_name}",
                    "system_message": f"{system_message}",
                    "operation": inspect.currentframe().f_code.co_name,
                },
            )

    @lru_cache(maxsize=1000)
    async def _resource_description(
        self, context_data: str = None, prompt: str = None
    ) -> str:
        """
        (async) Generates a resource description using a language model.
        Uses lru_cache with maxsize 1000 to cache the webpages data.

        :param context_data: (str) The context data to be used for generating the description. Defaults to None.
        :param prompt: (str) The prompt to be used for generating the description. Defaults to None.
        :returns str: The generated description with special characters removed.
        :raises ValueError: If neither prompt nor context_data is provided.
        :raises ConwayException: If there is an error generating the description.
        """
        try:
            if not prompt:
                if not context_data:
                    raise ValueError("Either prompt or context_data must be provided.")
                prompt = f"Craft a brief overview of summary of: {context_data}"
            description = await self.__model_chat(prompt)
            return remove_special_chars(description)
        except Exception as e:
            self.logger.error(f"Failed to generate LLM-based description: {e}")
            raise ConwayException(
                f"Failed to generate LLM-based description: {e}",
                context={
                    "context_data": f"{context_data}",
                    "prompt": f"{prompt}",
                    "operation": inspect.currentframe().f_code.co_name,
                },
            )


class BookmarkRetriever:
    """Handle bookmark retrieval from Chrome"""

    def __init__(
        self,
        logger: Optional[logging.Logger] = get_logger(
            os.environ.get("bookmarks_logger_name")
        ),
    ) -> None:
        self.logger = logger

    @lru_cache(maxsize=2000)
    def _retrieve_bookmarks(self) -> Generator:
        """
        Retrieves bookmarks from Chrome and yields them as a generator.
        Uses lru_cache with maxsize 2000 to cache the bookmarks.

        :returns Generator: A generator yielding bookmark data.
        :raises ConwayException: If there is an error retrieving the bookmarks.
        """
        try:
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
        except Exception as e:
            self.logger.error(f"Failed to retrieve bookmarks: {e}")
            raise ConwayException(
                f"Failed to retrieve bookmarks: {e}",
                context={"operation": inspect.currentframe().f_code.co_name},
            )


class WebsiteDescriptionFetcher:
    """Handle website content fetching and parsing"""

    def __init__(
        self,
        logger: logging.Logger = get_logger(os.environ.get("bookmarks_logger_name")),
    ) -> None:
        """
        Initializes the class with the specified logger.
        :param logger: (logging.Logger) The logger instance to be used.
        Defaults to a logger with the name specified in the environment
        variable 'bookmarks_logger_name'.
        """
        self.logger = logger

    @lru_cache(maxsize=2000)
    async def _get_website_description(
        self,
        url: str,
        timeout: Optional[int] = 5,
        headers: Optional[dict] = {},
        verify: Optional[bool] = False,
        max_redirects: Optional[int] = 5,
        follow_redirects: Optional[bool] = True,
        error_info: Optional[str] = "Page is not accessible",
    ) -> Tuple[str, str, int]:
        """
        (async) Fetch and parse the website description using httpx and BeautifulSoup.

        :param url: (str) The URL of the website to fetch the description from.
        :param timeout: (int, optional) The timeout for the HTTP request. Default is 5 seconds.
        :param headers: (dict, optional) Additional headers to send with the HTTP request.
        Default is an empty dictionary.
        :param verify: (bool, optional) Whether to verify the SSL certificate. Default is False.
        :param max_redirects: (int, optional) The maximum number of redirects to follow. Default is 5.
        :param follow_redirects: (bool, optional) Whether to follow redirects. Default is True.
        :param error_info: (str, optional) The error information to return in case of an error.
        Default is "Page is not accessible".
        :returns Tuple[str, str, int]: A tuple containing the title, description,
        and status code of the website.
        If an error occurs, the error_info string is returned for both the title and description.
        The status code is 420 in case of an error to distinguish it from regular status codes.
        """
        try:
            description = None
            if not headers:
                headers = get_random_headers()

            async with httpx.AsyncClient(
                follow_redirects=follow_redirects,
                max_redirects=max_redirects,
                verify=verify,
            ) as client:
                response = await client.get(url, timeout=timeout, headers=headers)

            status = int(response.status_code)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                title = soup.title.string if soup.title else "No title found"
                meta_description = soup.find("meta", attrs={"name": "description"})
                if meta_description:
                    description = meta_description.get("content")
                if not description:
                    main_domain = extract_domain(url)
                    if main_domain and main_domain != url:
                        return await self._get_website_description(main_domain)
            else:
                title = error_info
                description = error_info
            return title, description or "", status
        except Exception as e:
            self.logger.warning(f"Failed to fetch website description for {url}: {e}")
            return error_info, error_info, 420


class BookmarksOrganizer:
    """Handle database operations"""

    def __init__(
        self,
        model: Any,
        database: AbstractDatabase,
        bookmarks_retriever: Any,
        website_retriever: Any,
        logger: logging.Logger = get_logger(os.environ.get("bookmarks_logger_name")),
    ) -> None:
        """
        Initializes the class with the specified parameters.

        :param model: (Any) The model to be used.
        :param database: (AbstractDatabase) The database instance to be used.
        :param bookmarks_retriever: (Any) The bookmarks retriever instance to be used.
        :param website_retriever: (Any) The website retriever instance to be used.
        :param logger: (logging.Logger) The logger instance to be used. Defaults to
        a logger with the name specified in the environment variable 'bookmarks_logger_name'.
        """
        self.model = model
        self.database = database
        self.website_retriever = website_retriever
        self.bookmarks_retriever = bookmarks_retriever
        self.logger = logger

    async def _get_bookmarks(self, select: Optional[str] = None) -> set:
        """
        (async) Get all bookmarks stored in DB table.

        :param select: (str, opional) Select statement.
        :returns set: A set of unique bookmarks.
        :raises ConwayException: If there is an error in getting bookmarks from DB.
        """

        if not os.environ.get("db_table"):
            raise ValueError("DB table to store bookmarks data name is not provided.")

        all_urls = set()
        if not select:
            select: str = f"SELECT DISTINCT(url) FROM {os.environ.get('db_table')}"

        try:
            bookmarks_in_db = await self.database._select(select)
            self.logger.debug(f"Fetched {len(bookmarks_in_db)} bookmark(s) from DB")
            all_urls.update({row[0] for row in bookmarks_in_db})
            return all_urls
        except Exception as e:
            self.logger.error(f"Failed to fetch bookmarks from DB: {e}")
            raise ConwayException(
                f"Failed to fetch bookmarks from DB: {e}",
                context={
                    "all_urls": f"Number of fetched urls: {len(all_urls)}",
                    "select": f"{select}",
                    "operation": inspect.currentframe().f_code.co_name,
                },
            )

    def __verify_columns(self, columns: Optional[List[str] | str] = None) -> str:
        """
        Verify if provided columns are correct and return them as a string.

        :param columns: (list of str or a string, optional) Columns names to inserts.
        Either a prepared string or a list of strings. None by default. Can be provided via .env file.
        :returns str: Properly formatted columns string.
        :raises ConwayException: If there is an error in formatting columns data.
        """
        try:
            if not columns:
                if not os.environ.get("db_columns"):
                    raise ValueError(
                        "Columns must be provided in .env file or passed as parameter."
                    )
                columns: str = ",".join(
                    [str(c) for c in str(os.environ.get("db_columns")).split(",")]
                )
            else:
                if isinstance(columns, list):
                    columns: str = ",".join([str(c) for c in columns])
                else:
                    if not isinstance(columns, str):
                        raise ValueError(
                            "Columns must be a list of strings or a string."
                        )
            return columns
        except Exception as e:
            self.logger.error(f"Failed to process bookmarks: {e}")
            raise ConwayException(
                f"Failed to process bookmarks: {e}",
                context={
                    "columns_type": f"{type(columns)}",
                    "columns": f"{columns}",
                    "operation": inspect.currentframe().f_code.co_name,
                },
            )

    async def _insert_bookmarks(
        self, columns: Optional[List[str] | str] = None
    ) -> None:
        """
        (async) Insert prepared bookmarks into database.

        :param columns: Columns names to inserts. Either a prepared string or a list of strings.
        None by default. Can be provided via .env file.
        :raises ConwayException: If there is an error in inserting data.
        """

        columns = self.__verify_columns(columns)
        params: List[Tuple[str, str, str, str, str, str, str]] = []
        query = f"""INSERT INTO {os.environ.get("db_table")} ({columns}) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""

        all_bookmarks = await self._get_bookmarks()

        try:
            self.logger.debug("Processing bookmarks from Chrome")
            for item in self.bookmarks_retriever._retrieve_bookmarks():
                for url in item.get("urls", []):
                    if url.get("url") not in all_bookmarks:
                        title, content, status = (
                            await self.website_retriever._get_website_description(
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
                                await self.model._resource_description(
                                    context_data=f"{title} {content}"
                                ),
                                int(status),
                            )
                        )
                        all_bookmarks.add(url.get("url"))
            if params:
                self.logger.debug("Inserting bookmarks")
                try:
                    await self.database._insert_bulk(query, params)
                    self.logger.info("Bookmarks inserted successfully")
                except Exception as e:
                    await self.database._rollback()
                    self.logger.error(
                        f"Insert rollback. Failed to insert bookmarks: {e}"
                    )
            else:
                self.logger.debug("No new bookmarks to insert")
        except Exception as e:
            self.logger.error(f"Failed to fetch bookmarks from DB: {e}")
            raise ConwayException(
                f"Failed to fetch bookmarks from DB: {e}",
                context={
                    "all_bookmarks": f"Number of fetched urls: {len(all_bookmarks)}",
                    "query": f"{query}",
                    "operation": inspect.currentframe().f_code.co_name,
                },
            )
