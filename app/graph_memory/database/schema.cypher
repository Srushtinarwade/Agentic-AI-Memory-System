// Neo4j 2026.02+ / Cypher 25. Metadata filtering in vector indexes is required.
CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE;
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE;
CREATE INDEX entity_lookup IF NOT EXISTS FOR (e:Entity) ON (e.user_id, e.normalized_name, e.type);
CREATE INDEX fact_user_status IF NOT EXISTS FOR ()-[r:FACT]-() ON (r.user_id, r.status);
CREATE INDEX fact_subject_lookup IF NOT EXISTS FOR ()-[r:FACT]-() ON (r.user_id, r.subject_id, r.status);
CREATE INDEX fact_id_lookup IF NOT EXISTS FOR ()-[r:FACT]-() ON (r.fact_id, r.user_id);
CYPHER 25 CREATE VECTOR INDEX entity_vector_index IF NOT EXISTS
FOR (e:Entity) ON e.embedding WITH [e.user_id]
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};
CYPHER 25 CREATE VECTOR INDEX fact_vector_index IF NOT EXISTS
FOR ()-[r:FACT]-() ON r.embedding WITH [r.user_id, r.status]
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};
