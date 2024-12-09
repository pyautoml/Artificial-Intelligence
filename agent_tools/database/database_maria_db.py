import inspect
import logging
import aiomysql
import traceback
from typing import List, Tuple, Union, Self

# custom imports
from utilities.custom_exceptions import ConwayException
from database.database_abstract import AbstractDatabase
from utilities.custom_logger import get_logger, setup_logger


class MariaDB(AbstractDatabase):
    def __init__(
            self, 
            connection_settings: dict, 
            logger: logging.Logger = None, 
            logger_name: str = None, 
            log_level: str = None
        ) -> None:
        """
        Basic connection setting that is expected:
            {
                "host": "localhost", 
                "port": 3306, 
                "user": ..., 
                "password": ..., 
                "database": ...
            }

        :param logger: (logging.Logger) An existing logger instance.
        :param logger_name: (str) The name of the logger to be retrieved using `get_logger`.
        :param log_level: (str) The logging level to be used if a new logger is set up.
        """

        self.__pool = None
        self.logger = self.__set_logger(logger, logger_name, log_level)
        self.connection_settings = connection_settings
        self.ISOLATION_LEVELS = {
            'READ_UNCOMMITTED': "SET SESSION TRANSACTION ISOLATION LEVEL READ UNCOMMITTED",
            'READ_COMMITTED': "SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED",
            'REPEATABLE_READ': "SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ",
            'SERIALIZABLE': "SET SESSION TRANSACTION ISOLATION LEVEL SERIALIZABLE"
    }
        
    def __set_logger(
            self, 
            logger, 
            logger_name, 
            log_level
        ) -> logging.Logger:
        """
        Sets up the logger for the database operations.

        :param logger: (logging.Logger) An existing logger instance.
        :param logger_name: (str) The name of the logger to be retrieved using `get_logger`.
        :param log_level: (str) The logging level to be used if a new logger is set up.
        :raises ConwayException: If there is an error in setting up the logger.
        """
        try:
            if logger:
                self.logger = logger
            else:
                if logger_name:
                    self.logger = get_logger(logger_name)
                else:
                    self.logger = setup_logger("database", log_level=log_level or "INFO")
            return logger
        except Exception as e:
            self.logger.error(f"Failed to set logger: {e}")
            raise ConwayException(
                f"Failed to process bookmarks: {e}",
                context={
                    "logger": f"{logger}",
                    "logglogger_nameer": f"{logger_name}",
                    "log_level": f"{log_level}",
                    "operation": inspect.currentframe().f_code.co_name
                }
            )
        
    async def __aenter__(self) -> Self:
        """
        (async) Support for async context manager.
        :returns self: The instance of the class.
        """        
        await self._connect()
        return self

    async def __aexit__(self, exc_type: type, exc_val: Exception, exc_tb: traceback) -> None:
        """
        (async) Ensure proper cleanup when used as context manager.

        :param exc_type: (Exception) The exception type.
        :param exc_val: (Exception) The exception value.
        :param exc_tb: (traceback) The traceback object.
        """        
        await self._disconnect()

    async def _connect(
            self, 
            isolation_level: str = 'READ_COMMITTED', 
            pool_min_size: int = 1, 
            pool_max_size:int = 10, 
            autocommit: bool = False
            ) -> None:
        """
        (async) Establishes a connection pool to the MariaDB database.

        :param isolation_level: (str) The transaction isolation level. Defaults to 'READ_COMMITTED'.
        :param pool_min_size: (int) The minimum number of connections in the pool. Defaults to 1.
        :param pool_max_size: (int) The maximum number of connections in the pool. Defaults to 10.
        :param autocommit: (bool) Whether to autocommit transactions. Defaults to False.
        :raises ValueError: If the isolation level is not valid.
        :raises ConwayException: If there is an error establishing the connection pool.
        """
        try:
            if isolation_level not in self.ISOLATION_LEVELS:
                raise ValueError(f"Invalid isolation level. Choose from: {', '.join(self.ISOLATION_LEVELS.keys())}")

            init_command = self.ISOLATION_LEVELS[isolation_level]
            self.__pool = await aiomysql.create_pool(
                host=self.connection_settings["host"],
                port=int(self.connection_settings["port"]),
                user=self.connection_settings["user"],
                password=self.connection_settings["password"],
                db=self.connection_settings["database"],
                charset=self.connection_settings.get("charset", "utf8mb4") or "utf8mb4",
                minsize=pool_min_size,
                maxsize=pool_max_size,
                autocommit=autocommit,
                init_command=init_command
            )       
            self.logger.debug("Connection pool created successfully.")
        except KeyError as e:
            self.logger.error(f"Localhost connection missing a key in connection settings: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to establish localhost connection: {e}")
            raise ConwayException(
                f"Failed to process bookmarks: {e}",
                context={
                    "isolation_level": f"{isolation_level}",
                    "pool_min_size": f"{pool_min_size}",
                    "pool_max_size": f"{pool_max_size}",
                    "autocommit": f"{autocommit}",
                    "operation": inspect.currentframe().f_code.co_name
                }
            )


    async def _rollback(self) -> None:
        """
        (async) Rolls back the current transaction for the active connection.
        
        :raises ConwayException: If there is an error rolling back the transaction.
        """
        try:
            async with self.__pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("ROLLBACK")
                self.logger.debug("Transaction rolled back successfully.")
        except Exception as e:
            self.logger.error(f"Failed to rollback transaction: {e}")
            raise ConwayException(
                f"Failed to process bookmarks: {e}",
                context={
                    "operation": inspect.currentframe().f_code.co_name
                }
            )

    async def _disconnect(self) -> bool:
        """
        (async) Properly cleanup and close all database resources.

        :returns bool: True on success, False on failure.
        """
        try:
            if self.__pool:
                self.__pool.close()
                await self.__pool.wait_closed()
                self.logger.debug("Connection pool closed successfully.")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to cleanup database resources: {e}")
            return False
    
    # CRUD operations
    async def _select(self, query: str) -> List[tuple]:
        """
        (async) Executes a SELECT query using the connection pool.

        :param query: (str) The SELECT query to be executed.
        :returns: (List[tuple]) The result of the query as a list of tuples.
        :raises ConwayException: If there is an error executing the SELECT query.
        """
        try:
            async with self.__pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query)
                    result = await cur.fetchall()
                    self.logger.debug("SELECT query executed successfully.")
                    return result
        except Exception as e:
            self.logger.error(f"Failed to execute SELECT query: {e}")
            raise ConwayException(
                f"Failed to process bookmarks: {e}",
                context={
                    "query": f"{query}",
                    "operation": inspect.currentframe().f_code.co_name
                }
            )
        

    async def _insert(self, query: str, params: tuple, commit: bool = False) -> int|str:
        """
        (async) Executes an INSERT query and returns the inserted ID(s).

        :param query: (str) The INSERT query to be executed.
        :param params: (tuple) The parameters to be used in the query.
        :param commit: (bool) Whether to commit the transaction. Defaults to False.
        :returns int | str: The inserted ID(s).
        :raises ConwayException: If there is an error executing the INSERT query.
        """     
        try:
            async with self.__pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, params)
                    if commit:
                        await conn.commit()
                    inserted_id = cur.lastrowid
                    return inserted_id if inserted_id is not None else 0
        except Exception as e:
            self.logger.error(f"Failed to insert data. Query: {query}, Error: {e}")
            raise ConwayException(
                f"Failed to process bookmarks: {e}",
                context={
                    "query": f"{query}",
                    "params": f"{params}",
                    "commit": f"{commit}",
                    "operation": inspect.currentframe().f_code.co_name
                }
            )
        
    async def _insert_bulk(self, query: str, params: List[Tuple], commit: bool = True) -> List[Union[int, str]]:
        """
        (async) Executes a bulk INSERT query and returns the list of inserted IDs.

        :param query: (str) The INSERT query to be executed.
        :param params: (List[Tuple]) The parameters to be used in the query.
        :param commit: (bool) Whether to commit the transaction. Defaults to True.
        :returns List[Union[int, str]]: The list of inserted IDs.
        :raises ConwayException: If there is an error executing the bulk INSERT query.
        """
        try:
            async with self.__pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.executemany(query, params)
                    if commit:
                        await conn.commit()
                    first_inserted_id = cur.lastrowid
                    num_rows_inserted = cur.rowcount
                    inserted_ids = [first_inserted_id + i for i in range(num_rows_inserted)]
                    self.logger.debug(f"Bulk insert executed successfully. Inserted IDs: {inserted_ids}")
                    return inserted_ids
        except Exception as e:
            self.logger.error(f"Failed to bulk insert data. Query: {query}, Error: {e}")
            raise ConwayException(
                f"Failed to process bookmarks: {e}",
                context={
                    "query": f"{query}",
                    "commit": f"{commit}",
                    "operation": inspect.currentframe().f_code.co_name
                }
            )
                    
    async def _update(self, query: str, params: Tuple, commit: bool = True) -> int:
        """
        (async) Executes an UPDATE query and returns the number of affected rows.

        :param query: (str) The UPDATE query to be executed.
        :param params: (Tuple) The parameters to be used in the query.
        :param commit: (bool) Whether to commit the transaction. Defaults to True.
        :returns int: The number of affected rows.
        :raises ConwayException: If there is an error executing the UPDATE query.
        """
        try:
            async with self.__pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, params)
                    affected_rows = cur.rowcount
                    if commit:
                        await conn.commit()
                    self.logger.debug(f"Update executed successfully. Affected rows: {affected_rows}")
                    return affected_rows
        except Exception as e:
            self.logger.error(f"Failed to update data. Query: {query}, Error: {e}")
            raise ConwayException(
                f"Failed to process bookmarks: {e}",
                context={
                    "query": f"{query}",
                    "params": f"{params}",
                    "commit": f"{commit}",
                    "operation": inspect.currentframe().f_code.co_name
                }
            )
        
    async def _update_bulk(self, query: str, params: List[Tuple], commit: bool = True) -> int:
        """
        (async) Executes a bulk UPDATE query and returns the number of affected rows.

        :param query: (str) The UPDATE query to be executed.
        :param params: (List[Tuple]) The parameters to be used in the query.
        :param commit: (bool) Whether to commit the transaction. Defaults to True.
        :returns int: The number of affected rows.
        :raises ConwayException: If there is an error executing the bulk UPDATE query.
        """
        try:
            async with self.__pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.executemany(query, params)
                    if commit:
                        await conn.commit()
                    affected_rows = cur.rowcount
                    self.logger.debug(f"Bulk update executed successfully. Affected rows: {affected_rows}")
                    return affected_rows
        except Exception as e:
            self.logger.error(f"Failed to bulk update data. Query: {query}, Error: {e}")
            raise ConwayException(
                f"Failed to process bookmarks: {e}",
                context={
                    "query": f"{query}",
                    "commit": f"{commit}",
                    "operation": inspect.currentframe().f_code.co_name
                }
            )
                    
    async def _delete(self, query: str, params: Tuple, commit: bool = True) -> int:
        """
        (async) Executes a DELETE query and returns the number of affected rows.

        :param query: (str) The DELETE query to be executed.
        :param params: (Tuple) The parameters to be used in the query.
        :param commit: (bool) Whether to commit the transaction. Defaults to True.
        :returns int: The number of affected rows.
        :raises ConwayException: If there is an error executing the DELETE query.
        """
        try:
            async with self.__pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, params)
                    if commit:
                        await conn.commit()
                    affected_rows = cur.rowcount
                    self.logger.debug(f"Delete executed successfully. Affected rows: {affected_rows}")
                    return affected_rows
        except Exception as e:
            self.logger.error(f"Failed to delete data. Query: {query}, Error: {e}")
            raise ConwayException(
                f"Failed to process bookmarks: {e}",
                context={
                    "query": f"{query}",
                    "params": f"{params}",
                    "commit": f"{commit}",
                    "operation": inspect.currentframe().f_code.co_name
                }
            )
