-- Database initialization script for Pharmacogenomics ML Platform
-- Creates necessary databases and extensions

-- Create MLflow database for experiment tracking
CREATE DATABASE mlflow;

-- Grant permissions to the application user
GRANT ALL PRIVILEGES ON DATABASE pharmacogenomics TO pgx_user;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO pgx_user;

-- Connect to the main database
\c pharmacogenomics;

-- Create necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create schema for different components
CREATE SCHEMA IF NOT EXISTS data_ingestion;
CREATE SCHEMA IF NOT EXISTS quality_scoring;
CREATE SCHEMA IF NOT EXISTS ml_models;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Grant schema permissions
GRANT ALL ON SCHEMA data_ingestion TO pgx_user;
GRANT ALL ON SCHEMA quality_scoring TO pgx_user;
GRANT ALL ON SCHEMA ml_models TO pgx_user;
GRANT ALL ON SCHEMA analytics TO pgx_user;

-- Connect to MLflow database and set up
\c mlflow;

-- Grant permissions for MLflow
GRANT ALL PRIVILEGES ON DATABASE mlflow TO pgx_user;
GRANT ALL ON SCHEMA public TO pgx_user;