"""Database connection management and session handling."""

import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from ..config import get_config, get_logger
from .models import Base


logger = get_logger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self, config=None):
        """Initialize database manager.
        
        Args:
            config: Database configuration. If None, loads from global config.
        """
        if config is None:
            app_config = get_config()
            config = app_config.database
        
        self.config = config
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize database engine and session factory."""
        if self._initialized:
            return
        
        logger.info(f"Initializing database connection to {self._get_safe_url()}")
        
        # Create engine with configuration
        engine_kwargs = {
            'pool_size': self.config.pool_size,
            'max_overflow': self.config.max_overflow,
            'pool_timeout': self.config.pool_timeout,
            'pool_recycle': self.config.pool_recycle,
            'echo': self.config.echo,
        }
        
        # Handle SQLite-specific configuration
        if self.config.url.startswith('sqlite'):
            # SQLite doesn't support connection pooling in the same way
            engine_kwargs = {
                'poolclass': StaticPool,
                'connect_args': {'check_same_thread': False},
                'echo': self.config.echo
            }
        
        self.engine = create_engine(self.config.url, **engine_kwargs)
        
        # Add connection event listeners
        self._setup_event_listeners()
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        self._initialized = True
        logger.info("Database connection initialized successfully")
    
    def _setup_event_listeners(self) -> None:
        """Setup database event listeners for monitoring and optimization."""
        
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set SQLite pragmas for better performance."""
            if self.config.url.startswith('sqlite'):
                cursor = dbapi_connection.cursor()
                # Enable foreign key constraints
                cursor.execute("PRAGMA foreign_keys=ON")
                # Set journal mode for better concurrency
                cursor.execute("PRAGMA journal_mode=WAL")
                # Set synchronous mode for better performance
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.close()
        
        @event.listens_for(self.engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Log slow queries for performance monitoring."""
            context._query_start_time = logger._get_time() if hasattr(logger, '_get_time') else None
        
        @event.listens_for(self.engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Log completed queries and performance metrics."""
            if hasattr(context, '_query_start_time') and context._query_start_time:
                total = logger._get_time() - context._query_start_time if hasattr(logger, '_get_time') else 0
                if total > 1.0:  # Log queries taking more than 1 second
                    logger.warning(f"Slow query detected: {total:.2f}s - {statement[:100]}...")
    
    def create_tables(self) -> None:
        """Create all database tables."""
        if not self._initialized:
            self.initialize()
        
        logger.info("Creating database tables")
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")
    
    def drop_tables(self) -> None:
        """Drop all database tables. Use with caution!"""
        if not self._initialized:
            self.initialize()
        
        logger.warning("Dropping all database tables")
        Base.metadata.drop_all(bind=self.engine)
        logger.info("Database tables dropped")
    
    def get_session(self) -> Session:
        """Get a new database session.
        
        Returns:
            Session: SQLAlchemy session instance
        """
        if not self._initialized:
            self.initialize()
        
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations.
        
        Yields:
            Session: SQLAlchemy session with automatic commit/rollback
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            session.close()
    
    def health_check(self) -> bool:
        """Check database connectivity and health.
        
        Returns:
            bool: True if database is healthy, False otherwise
        """
        try:
            with self.session_scope() as session:
                # Simple query to test connectivity
                session.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False
    
    def get_connection_info(self) -> dict:
        """Get database connection information.
        
        Returns:
            dict: Connection information and statistics
        """
        if not self._initialized:
            return {"status": "not_initialized"}
        
        info = {
            "status": "initialized",
            "url": self._get_safe_url(),
            "pool_size": getattr(self.engine.pool, 'size', None),
            "checked_out": getattr(self.engine.pool, 'checkedout', None),
            "overflow": getattr(self.engine.pool, 'overflow', None),
            "checked_in": getattr(self.engine.pool, 'checkedin', None),
        }
        
        return info
    
    def _get_safe_url(self) -> str:
        """Get database URL with credentials masked for logging.
        
        Returns:
            str: Safe database URL for logging
        """
        url = self.config.url
        if '@' in url:
            # Mask credentials in URL
            parts = url.split('@')
            if len(parts) == 2:
                protocol_and_creds = parts[0]
                host_and_db = parts[1]
                if '://' in protocol_and_creds:
                    protocol = protocol_and_creds.split('://')[0]
                    return f"{protocol}://***:***@{host_and_db}"
        return url
    
    def execute_bulk_insert(self, model_class, data_list: list) -> int:
        """Execute bulk insert operation for better performance.
        
        Args:
            model_class: SQLAlchemy model class
            data_list: List of dictionaries with data to insert
            
        Returns:
            int: Number of records inserted
        """
        if not data_list:
            return 0
        
        try:
            with self.session_scope() as session:
                session.bulk_insert_mappings(model_class, data_list)
                return len(data_list)
        except Exception as e:
            logger.error(f"Bulk insert failed: {str(e)}")
            raise
    
    def execute_bulk_update(self, model_class, data_list: list) -> int:
        """Execute bulk update operation for better performance.
        
        Args:
            model_class: SQLAlchemy model class
            data_list: List of dictionaries with data to update
            
        Returns:
            int: Number of records updated
        """
        if not data_list:
            return 0
        
        try:
            with self.session_scope() as session:
                session.bulk_update_mappings(model_class, data_list)
                return len(data_list)
        except Exception as e:
            logger.error(f"Bulk update failed: {str(e)}")
            raise
    
    def close(self) -> None:
        """Close database connections and cleanup resources."""
        if self.engine:
            logger.info("Closing database connections")
            self.engine.dispose()
            self.engine = None
            self.SessionLocal = None
            self._initialized = False


# Global database manager instance
_db_manager = DatabaseManager()


def init_database(config=None) -> None:
    """Initialize the global database manager.
    
    Args:
        config: Database configuration. If None, loads from global config.
    """
    global _db_manager
    if config:
        _db_manager = DatabaseManager(config)
    _db_manager.initialize()


def create_tables() -> None:
    """Create all database tables using the global manager."""
    _db_manager.create_tables()


def get_db_session() -> Session:
    """Get a database session from the global manager.
    
    Returns:
        Session: SQLAlchemy session instance
    """
    return _db_manager.get_session()


@contextmanager
def db_session_scope() -> Generator[Session, None, None]:
    """Get a transactional database session scope.
    
    Yields:
        Session: SQLAlchemy session with automatic commit/rollback
    """
    with _db_manager.session_scope() as session:
        yield session


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance.
    
    Returns:
        DatabaseManager: Global database manager
    """
    return _db_manager