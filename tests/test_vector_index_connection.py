from ari_llm_platform import VectorIndex


def test_connect_loads_sqlite_vec(tmp_path):
    database_path = tmp_path / "ari.db"

    vector_index = VectorIndex()
    vector_index.initialize(str(database_path))

    connection = vector_index.connect()
    try:
        vec_version = connection.execute(
            "SELECT vec_version()"
        ).fetchone()[0]

        assert vec_version
        assert database_path.exists()
    finally:
        vector_index.close()