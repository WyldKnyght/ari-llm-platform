from ari_llm_platform import VectorIndex, migrate_phase3, rollback_phase3


def test_phase3_schema_migration_creates_core_tables(tmp_path):
    database_path = tmp_path / "ari.db"

    vector_index = VectorIndex()
    vector_index.initialize(str(database_path))
    connection = vector_index.connect()

    try:
        migrate_phase3(connection)

        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

        assert {
            "documents",
            "chunks",
            "tags",
            "document_tags",
            "chunk_embeddings",
        }.issubset(table_names)

        document_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(documents)")
        }

        assert {
            "source_type",
            "source_ref",
            "title",
            "metadata",
        }.issubset(document_columns)

        migrate_phase3(connection)
    finally:
        vector_index.close()


def test_phase3_schema_rollback_removes_core_tables(tmp_path):
    database_path = tmp_path / "ari.db"

    vector_index = VectorIndex()
    vector_index.initialize(str(database_path))
    connection = vector_index.connect()

    try:
        migrate_phase3(connection)
        rollback_phase3(connection)

        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

        assert "documents" not in table_names
        assert "chunks" not in table_names
        assert "tags" not in table_names
        assert "document_tags" not in table_names
        assert "chunk_embeddings" not in table_names
    finally:
        vector_index.close()