import pandas as pd
from pathlib import Path


class TableLoader:

    def load(self, file_path):

        file_path = Path(file_path)

        extension = file_path.suffix.lower()

        print("\n" + "=" * 60)
        print("TABLE LOADER")
        print("=" * 60)

        print(f"File : {file_path}")
        print(f"Extension : {extension}")

        # =====================================================
        # CSV
        # =====================================================

        if extension == ".csv":

            df = pd.read_csv(
                file_path
            )

        # =====================================================
        # EXCEL
        # =====================================================

        elif extension in [
            ".xlsx",
            ".xls"
        ]:

            df = pd.read_excel(
                file_path
            )

        else:

            raise ValueError(
                f"Unsupported table format: {extension}"
            )

        print(
            f"Rows : {len(df)}"
        )

        print(
            f"Columns : {len(df.columns)}"
        )

        print(
            f"Column Names : {list(df.columns)}"
        )

        if df.empty:

            print("WARNING: Table is empty!")

            return []

        documents = []

        # =====================================================
        # CONVERT EACH ROW TO TEXT
        # =====================================================

        for index, row in df.iterrows():

            lines = []

            lines.append(
                f"Source File: {file_path.name}"
            )

            for column in df.columns:

                value = row[column]

                if pd.isna(value):

                    value = ""

                lines.append(
                    f"{column}: {value}"
                )

            text = "\n".join(lines)

            documents.append(text)

        print(
            f"Documents Created : {len(documents)}"
        )

        # =====================================================
        # SHOW FIRST ROW
        # =====================================================

        if documents:

            print("\nFirst Row:")

            print("-" * 60)

            print(
                documents[0]
            )

            print("-" * 60)

        return documents