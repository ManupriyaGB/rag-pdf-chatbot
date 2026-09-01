import pandas as pd
import re
from pathlib import Path


class TableLoader:

    def __init__(self, data_dir="data"):

        self.data_dir = Path(data_dir)

        self.tables = {}

        print("=" * 60)
        print("LOADING TABLE DATA")
        print("=" * 60)

        self.load_tables()

        print(
            f"Tables Loaded : {len(self.tables)}"
        )


    # =========================================================
    # LOAD CSV + XLSX
    # =========================================================

    def load_tables(self):

        if not self.data_dir.exists():

            print(
                f"Data directory not found: "
                f"{self.data_dir}"
            )

            return


        for file in self.data_dir.rglob("*"):

            if not file.is_file():
                continue


            suffix = file.suffix.lower()


            try:

                # ------------------------------------------------
                # CSV
                # ------------------------------------------------

                if suffix == ".csv":

                    df = pd.read_csv(file)

                    self.tables[file.name] = df

                    print(
                        f"Loaded CSV : "
                        f"{file.name} "
                        f"({len(df)} rows)"
                    )


                # ------------------------------------------------
                # Excel
                # ------------------------------------------------

                elif suffix in [".xlsx", ".xls"]:

                    excel = pd.ExcelFile(file)

                    for sheet in excel.sheet_names:

                        df = pd.read_excel(
                            file,
                            sheet_name=sheet
                        )

                        key = (
                            f"{file.name}"
                            f"::{sheet}"
                        )

                        self.tables[key] = df

                        print(
                            f"Loaded Excel : "
                            f"{key} "
                            f"({len(df)} rows)"
                        )


            except Exception as e:

                print(
                    f"ERROR loading "
                    f"{file}: {e}"
                )


    # =========================================================
    # GET ALL TABLES
    # =========================================================

    def get_tables(self):

        return self.tables


    # =========================================================
    # SEARCH TABLE
    # =========================================================

    def search(self, query):

        query_lower = query.lower()

        results = []


        for table_name, df in self.tables.items():

            for index, row in df.iterrows():

                row_text = self.row_to_text(
                    row
                )

                if self.row_matches_text(
                    row_text,
                    query_lower
                ):

                    results.append(
                        {
                            "table": table_name,
                            "row": index,
                            "text": row_text
                        }
                    )


        return results


    # =========================================================
    # ROW → TEXT
    # =========================================================

    def row_to_text(
        self,
        row
    ):

        values = []

        for column in row.index:

            value = row[column]

            if pd.isna(value):

                continue

            values.append(
                f"{column}: {value}"
            )

        return " | ".join(values)


    # =========================================================
    # TEXT MATCH
    # =========================================================

    def row_matches_text(
        self,
        row_text,
        query
    ):

        words = query.split()

        meaningful_words = [

            word
            for word in words
            if len(word) > 2

        ]

        if not meaningful_words:

            return False


        row_lower = row_text.lower()


        matches = sum(

            word in row_lower

            for word in meaningful_words

        )


        return matches >= 1


    # =========================================================
    # LOCATION SEARCH
    # =========================================================

    def search_location(
        self,
        location
    ):

        results = []


        for table_name, df in self.tables.items():

            for index, row in df.iterrows():

                for column in df.columns:

                    value = str(
                        row[column]
                    ).strip().lower()


                    if value == location.lower():

                        results.append(
                            {
                                "table": table_name,
                                "row": index,
                                "text":
                                    self.row_to_text(row)
                            }
                        )

                        break


        return results


    # =========================================================
    # NUMERIC CONDITION SEARCH
    # =========================================================

    def search_numeric(
        self,
        column_name,
        operator,
        value
    ):

        results = []


        for table_name, df in self.tables.items():

            matching_columns = [

                column
                for column in df.columns

                if column_name.lower()
                in str(column).lower()

            ]


            for column in matching_columns:

                numeric_values = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )


                if operator == "<":

                    mask = (
                        numeric_values
                        < value
                    )


                elif operator == ">":

                    mask = (
                        numeric_values
                        > value
                    )


                elif operator == "<=":

                    mask = (
                        numeric_values
                        <= value
                    )


                elif operator == ">=":

                    mask = (
                        numeric_values
                        >= value
                    )


                elif operator == "=":

                    mask = (
                        numeric_values
                        == value
                    )


                else:

                    continue


                matched_df = df[mask]


                for index, row in matched_df.iterrows():

                    results.append(
                        {
                            "table": table_name,
                            "row": index,
                            "text":
                                self.row_to_text(row)
                        }
                    )


        return results