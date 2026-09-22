#!/bin/bash
# Clears all ingested document chunks so you can re-demo ingestion from a
# clean state. Does NOT touch containers, the schema, or the vector
# extension -- just empties the table.
docker exec -it rag-postgres psql -U postgres -d rag_db -c "TRUNCATE TABLE data_employee_handbook;"
echo "Knowledge base cleared. Ready to re-ingest."