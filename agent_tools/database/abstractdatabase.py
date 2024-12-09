
from abc import ABC, abstractmethod
from typing import List, Tuple, Union


class AbstractDatabase(ABC):

    @abstractmethod
    async def _connect( 
        self, 
        isolation_level: str = 'READ_COMMITTED', 
        pool_min_size: int = 1, 
        pool_max_size:int = 10, 
        autocommit: bool = False) -> None:
        """
        (async) Establishes a connection pool to the MariaDB database.

        :param isolation_level: (str) The transaction isolation level. Defaults to 'READ_COMMITTED'.
        :param pool_min_size: (int) The minimum number of connections in the pool. Defaults to 1.
        :param pool_max_size: (int) The maximum number of connections in the pool. Defaults to 10.
        :param autocommit: (bool) Whether to autocommit transactions. Defaults to False.
        """
        pass

    @abstractmethod
    async def _disconnect(self) -> bool:
        """
        (async) Properly cleanup and close all database resources.
        :returns bool: True on success, False on failure.
        """
        pass
    
    @abstractmethod
    async def _select(self, query: str) -> List[tuple]:
        """
        (async) Executes a SELECT query using the connection pool.

        :param query: (str) The SELECT query to be executed.
        :returns: (List[tuple]) The result of the query as a list of tuples.
        """
        pass
    
    @abstractmethod
    async def _insert(
        self, 
        query: str, 
        params: tuple, 
        commit: bool = False
    ) -> int|str:
        """
        (async) Executes an INSERT query and returns the inserted ID(s).

        :param query: (str) The INSERT query to be executed.
        :param params: (tuple) The parameters to be used in the query.
        :param commit: (bool) Whether to commit the transaction. Defaults to False.
        :returns int | str: The inserted ID(s).
        """   
        pass

    @abstractmethod
    async def _insert_bulk(
        self, 
        query: str, 
        params: List[Tuple], 
        commit: bool = True
    ) -> List[Union[int, str]]:
        """
        (async) Executes a bulk INSERT query and returns the list of inserted IDs.

        :param query: (str) The INSERT query to be executed.
        :param params: (List[Tuple]) The parameters to be used in the query.
        :param commit: (bool) Whether to commit the transaction. Defaults to True.
        :returns List[Union[int, str]]: The list of inserted IDs.
        """
        pass

    @abstractmethod
    async def _update(self) -> int:
        """
        (async) Executes an UPDATE query and returns the number of affected rows.

        :param query: (str) The UPDATE query to be executed.
        :param params: (Tuple) The parameters to be used in the query.
        :param commit: (bool) Whether to commit the transaction. Defaults to True.
        :returns int: The number of affected rows.
        """
        pass

    @abstractmethod
    async def _update_bulk(
        self, 
        query: str, 
        params: List[Tuple], 
        commit: bool = True
    ) -> int:
        """
        (async) Executes a bulk UPDATE query and returns the number of affected rows.

        :param query: (str) The UPDATE query to be executed.
        :param params: (List[Tuple]) The parameters to be used in the query.
        :param commit: (bool) Whether to commit the transaction. Defaults to True.
        :returns int: The number of affected rows.
        """
        pass

    @abstractmethod
    async def _delete(
        self, 
        query: str, 
        params: Tuple, 
        commit: bool = True
    ) -> int:
        """
        (async) Executes a DELETE query and returns the number of affected rows.

        :param query: (str) The DELETE query to be executed.
        :param params: (Tuple) The parameters to be used in the query.
        :param commit: (bool) Whether to commit the transaction. Defaults to True.
        :returns int: The number of affected rows.
        """
        pass

    @abstractmethod
    async def _rollback(self) -> None:
        """(async) Rolls back the current transaction for the active connection."""
        pass
