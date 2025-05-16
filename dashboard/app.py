import seaborn as sns
from faicons import icon_svg

# Import data from shared.py
from shared import app_dir, df

from functools import partial
from shiny import reactive
from shiny.express import input, render, ui
from shiny.ui import page_navbar


from functools import partial

from shiny.express import ui
from shiny.ui import page_navbar

ui.page_opts(
    title="App with navbar",  
    page_fn=partial(page_navbar, id="page"),  
)

with ui.nav_panel("A"):  
    "Page A content"

with ui.nav_panel("B"):  
    "Page B content"

with ui.nav_panel("C"):  
    "Page C content"
# with ui.navset_tab():
#     with ui.nav_panel("A"):
#         ui.h2("Page A content")

#     with ui.nav_panel("B"):
#         ui.h2("Page B content")

#     with ui.nav_panel("C"):
#         ui.h2("Page C content")

with ui.sidebar(title="Filter controls"):
    ui.input_slider("mass", "Mass", 2000, 6000, 6000)
    ui.input_checkbox_group(
        "species",
        "Species",
        ["Adelie", "Gentoo", "Chinstrap"],
        selected=["Adelie", "Gentoo", "Chinstrap"],
    )


with ui.layout_column_wrap(fill=False):
    with ui.value_box(showcase=icon_svg("earlybirds")):
        "Number of penguins"

        @render.text
        def count():
            return filtered_df().shape[0]

    with ui.value_box(showcase=icon_svg("ruler-horizontal")):
        "Average bill length"

        @render.text
        def bill_length():
            return f"{filtered_df()['bill_length_mm'].mean():.1f} mm"

    with ui.value_box(showcase=icon_svg("ruler-vertical")):
        "Average bill depth"

        @render.text
        def bill_depth():
            return f"{filtered_df()['bill_depth_mm'].mean():.1f} mm"


with ui.layout_columns():
    with ui.card(full_screen=True):
        ui.card_header("Bill length and depth")

        @render.plot
        def length_depth():
            return sns.scatterplot(
                data=filtered_df(),
                x="bill_length_mm",
                y="bill_depth_mm",
                hue="species",
            )

    with ui.card(full_screen=True):
        ui.card_header("Penguin data")

        @render.data_frame
        def summary_statistics():
            cols = [
                "species",
                "island",
                "bill_length_mm",
                "bill_depth_mm",
                "body_mass_g",
            ]
            return render.DataGrid(filtered_df()[cols], filters=True)


ui.include_css(app_dir / "styles.css")


@reactive.calc
def filtered_df():
    filt_df = df[df["species"].isin(input.species())]
    filt_df = filt_df.loc[filt_df["body_mass_g"] < input.mass()]
    return filt_df
