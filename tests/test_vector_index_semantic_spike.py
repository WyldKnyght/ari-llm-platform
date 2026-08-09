from sqlite_vec import serialize_float32

from ari_llm_platform import VectorIndex


def test_sqlite_vec_returns_expected_nearest_neighbor(tmp_path):
    database_path = tmp_path / "semantic_spike.db"

    vector_index = VectorIndex()
    vector_index.initialize(str(database_path))
    connection = vector_index.connect()

    try:
        connection.execute(
            """
            CREATE VIRTUAL TABLE semantic_spike_vectors
            USING vec0(
                embedding float[3]
            )
            """
        )

        connection.executemany(
            """
            INSERT INTO semantic_spike_vectors (rowid, embedding)
            VALUES (?, ?)
            """,
            [
                (1, serialize_float32([0.0, 0.0, 0.0])),
                (2, serialize_float32([1.0, 1.0, 1.0])),
                (3, serialize_float32([0.0, 1.0, 0.0])),
            ],
        )

        nearest_neighbor = connection.execute(
            """
            SELECT rowid, distance
            FROM semantic_spike_vectors
            WHERE embedding MATCH ?
              AND k = 1
            """,
            [serialize_float32([0.1, 0.0, 0.0])],
        ).fetchone()

        assert nearest_neighbor is not None
        assert nearest_neighbor[0] == 1
        assert nearest_neighbor[1] < 0.2
    finally:
        vector_index.close()