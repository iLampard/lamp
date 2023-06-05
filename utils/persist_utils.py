


class PersistDB:
    def __init__(
            self,
            storage_uri: str = '.ddb_storage',
            use_compression: bool = True,
    ):
        import dictdatabase
        self._db = dictdatabase

        self._db.config.storage_directory = storage_uri
        self._db.config.use_compression = use_compression

    @property
    def db(self):
        return self._db
