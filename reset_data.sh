
docker exec -it rag-postgres psql -U postgres -d rag_db -c "TRUNCATE TABLE data_employee_handbook;"
echo "Knowledge base cleared. Ready to re-ingest."