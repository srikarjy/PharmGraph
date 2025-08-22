"""Tests for database connection management."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.storage.database import DatabaseManager, init_database, get_db_session
from src.storage.models import Base, PaperModel
from src.config.models import DatabaseConfig


class TestDatabaseManager:
    """Test cases for DatabaseManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = Path(self.temp_dir) / "test.db"
        
        self.config = DatabaseConfig(
            url=f"sqlite:///{self.test_db_path}",
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600,
            echo=False
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_database_manager_initialization(self):
        """Test database manager initialization."""
        db_manager = DatabaseManager(self.config)
        
        assert db_manager.config == self.config
        assert db_manager.engine is None
        assert db_manager.SessionLocal is None
        assert not db_manager._initialized
    
    def test_database_initialization(self):
        """Test database initialization."""
        db_manager = DatabaseManager(self.config)
        db_manager.initialize()
        
        assert db_manager._initialized
        assert db_manager.engine is not None
        assert db_manager.SessionLocal is not None
    
    def test_create_tables(self):
        """Test table creation."""
        db_manager = DatabaseManager(self.config)
        db_manager.create_tables()
        
        # Verify tables were created by checking if we can create a session
        session = db_manager.get_session()
        assert session is not None
        
        # Try to query a table (should not raise an error)
        result = session.query(PaperModel).count()
        assert result == 0
        
        session.close()
    
    def test_session_creation(self):
        """Test database session creation."""
        db_manager = DatabaseManager(self.config)
        db_manager.initialize()
        
        session = db_manager.get_session()
        assert session is not None
        
        # Test that we can use the session
        paper = PaperModel(
            pmid='test_paper_1',
            title='Test Paper for Database Connection'
        )
        
        session.add(paper)
        session.commit()
        
        # Verify the paper was saved
        saved_paper = session.query(PaperModel).filter_by(pmid='test_paper_1').first()
        assert saved_paper is not None
        assert saved_paper.title == 'Test Paper for Database Connection'
        
        session.close()
    
    def test_session_scope_context_manager(self):
        """Test session scope context manager."""
        db_manager = DatabaseManager(self.config)
        db_manager.create_tables()
        
        # Test successful transaction
        with db_manager.session_scope() as session:
            paper = PaperModel(
                pmid='test_paper_2',
                title='Test Paper for Session Scope'
            )
            session.add(paper)
        
        # Verify the paper was committed
        with db_manager.session_scope() as session:
            saved_paper = session.query(PaperModel).filter_by(pmid='test_paper_2').first()
            assert saved_paper is not None
    
    def test_session_scope_rollback_on_error(self):
        """Test session scope rollback on error."""
        db_manager = DatabaseManager(self.config)
        db_manager.create_tables()
        
        # Test transaction rollback on error
        with pytest.raises(ValueError):
            with db_manager.session_scope() as session:
                paper = PaperModel(
                    pmid='test_paper_3',
                    title='Test Paper for Rollback'
                )
                session.add(paper)
                session.flush()  # Force the insert
                
                # Raise an error to trigger rollback
                raise ValueError("Test error")
        
        # Verify the paper was not saved due to rollback
        with db_manager.session_scope() as session:
            saved_paper = session.query(PaperModel).filter_by(pmid='test_paper_3').first()
            assert saved_paper is None
    
    def test_health_check(self):
        """Test database health check."""
        db_manager = DatabaseManager(self.config)
        db_manager.initialize()
        
        # Health check should pass for a working database
        assert db_manager.health_check() is True
    
    def test_health_check_failure(self):
        """Test health check failure."""
        # Create a manager with invalid database URL
        invalid_config = DatabaseConfig(url="sqlite:///nonexistent/path/test.db")
        db_manager = DatabaseManager(invalid_config)
        
        # Health check should fail
        assert db_manager.health_check() is False
    
    def test_connection_info(self):
        """Test connection information retrieval."""
        db_manager = DatabaseManager(self.config)
        
        # Before initialization
        info = db_manager.get_connection_info()
        assert info['status'] == 'not_initialized'
        
        # After initialization
        db_manager.initialize()
        info = db_manager.get_connection_info()
        assert info['status'] == 'initialized'
        assert 'url' in info
        assert 'sqlite' in info['url']
    
    def test_safe_url_masking(self):
        """Test URL credential masking."""
        # Test URL with credentials
        config_with_creds = DatabaseConfig(url="postgresql://user:password@localhost:5432/testdb")
        db_manager = DatabaseManager(config_with_creds)
        
        safe_url = db_manager._get_safe_url()
        assert 'password' not in safe_url
        assert '***:***' in safe_url
        assert 'postgresql' in safe_url
        assert 'localhost:5432/testdb' in safe_url
    
    def test_close_connections(self):
        """Test closing database connections."""
        db_manager = DatabaseManager(self.config)
        db_manager.initialize()
        
        assert db_manager._initialized
        assert db_manager.engine is not None
        
        db_manager.close()
        
        assert not db_manager._initialized
        assert db_manager.engine is None
        assert db_manager.SessionLocal is None
    
    def test_postgresql_configuration(self):
        """Test PostgreSQL-specific configuration."""
        pg_config = DatabaseConfig(
            url="postgresql://user:pass@localhost:5432/testdb",
            pool_size=20,
            max_overflow=30,
            pool_timeout=60
        )
        
        db_manager = DatabaseManager(pg_config)
        
        # Should not raise an error during initialization setup
        # (even though connection will fail in test environment)
        assert db_manager.config.pool_size == 20
        assert db_manager.config.max_overflow == 30


class TestGlobalDatabaseFunctions:
    """Test cases for global database functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = Path(self.temp_dir) / "global_test.db"
        
        self.config = DatabaseConfig(
            url=f"sqlite:///{self.test_db_path}",
            echo=False
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Reset global database manager
        from src.storage.database import _db_manager
        if _db_manager.engine:
            _db_manager.close()
    
    def test_init_database_global(self):
        """Test global database initialization."""
        init_database(self.config)
        
        # Should be able to get a session
        session = get_db_session()
        assert session is not None
        session.close()
    
    def test_global_session_scope(self):
        """Test global session scope context manager."""
        from src.storage.database import db_session_scope, create_tables
        
        init_database(self.config)
        create_tables()
        
        # Test using global session scope
        with db_session_scope() as session:
            paper = PaperModel(
                pmid='global_test_paper',
                title='Global Test Paper'
            )
            session.add(paper)
        
        # Verify the paper was saved
        with db_session_scope() as session:
            saved_paper = session.query(PaperModel).filter_by(pmid='global_test_paper').first()
            assert saved_paper is not None
            assert saved_paper.title == 'Global Test Paper'
    
    def test_get_database_manager(self):
        """Test getting global database manager."""
        from src.storage.database import get_database_manager
        
        manager = get_database_manager()
        assert manager is not None
        
        # Should be the same instance on subsequent calls
        manager2 = get_database_manager()
        assert manager is manager2