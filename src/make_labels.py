"""
Write data/label_provenance.csv — the documented Accra flood hotspots that
anchor the positive class.

IMPORTANT HONESTY NOTE (also stated in the README): these are *documented
flood-prone neighborhoods* compiled from peer-reviewed literature and news
reporting — NOT a formal, field-validated flood inventory. Coordinates are
neighborhood centroids from a geocoder (Glefe/Dansoman marked approximate).
The model therefore learns "documented hotspot likeness", and results must
be read as a susceptibility screening, not an engineering-grade flood map.
"""
import csv
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "label_provenance.csv")

SRC_OWUSU = ("Owusu et al. (2019), via 'Urban Flooding, Adaptation Strategies, and "
             "Resilience: Case Study of Accra, Ghana' (Springer). Lists these as "
             "neighborhoods prone to perennial flooding.")
SRC_HUMAN_FACTOR = ("'Perennial Flooding in the Accra Metropolis - The Human Factor' "
                    "(research report). Describes these communities on the lower "
                    "Odaw floodplain where flooding is endemic.")
SRC_2015 = ("June 3, 2015 Accra floods: Kwame Nkrumah Circle / Odawna disaster, "
            "widely documented (e.g., Wikipedia '2015 Accra floods'; Ghana news media).")
SRC_2022 = ("BusinessGhana reporting on May 2022 Accra floods: areas inundated "
            "included Odawna, Abossey Okai, Kaneshie, Dansoman, Kwame Nkrumah "
            "Interchange, Alajo, Agbogbloshie.")
SRC_2026 = ("Accra Street Journal, Aug 2026, on June 2026 Accra floods: Kaneshie, "
            "Odawna, Adabraka, Circle, Lapaz, Alajo, Mallam submerged.")

# (name, lon, lat, source, coordinate_quality)
HOTSPOTS = [
    ("Odawna", -0.22701, 5.56932, SRC_OWUSU + " Also " + SRC_2015, "neighborhood centroid"),
    ("Kaneshie", -0.24492, 5.57641, SRC_OWUSU + " Also " + SRC_2022 + " " + SRC_2026, "neighborhood centroid"),
    ("Adabraka", -0.21068, 5.56409, SRC_OWUSU + " Also " + SRC_HUMAN_FACTOR + " " + SRC_2026, "neighborhood centroid"),
    ("Alajo", -0.21137, 5.59771, SRC_OWUSU + " Also " + SRC_HUMAN_FACTOR + " " + SRC_2022, "neighborhood centroid"),
    ("Mataheko", -0.23558, 5.55941, SRC_OWUSU, "neighborhood centroid"),
    ("Nima", -0.19989, 5.58390, SRC_OWUSU, "neighborhood centroid"),
    ("Mamobi", -0.19988, 5.59971, SRC_OWUSU, "neighborhood centroid"),
    ("Abelemkpe", -0.21856, 5.60625, SRC_OWUSU, "neighborhood centroid"),
    ("Asylum Down", -0.20632, 5.56919, SRC_OWUSU, "neighborhood centroid"),
    ("Agbogbloshie", -0.21435, 5.55346, SRC_2022, "neighborhood centroid"),
    ("Abossey Okai", -0.22393, 5.55338, SRC_2022, "neighborhood centroid"),
    ("Dzorwulu", -0.20347, 5.61053, SRC_HUMAN_FACTOR, "neighborhood centroid"),
    ("Kotobabi", -0.20084, 5.59580, SRC_HUMAN_FACTOR, "neighborhood centroid"),
    ("Kwame Nkrumah Circle", -0.20614, 5.58555, SRC_2015 + " Also " + SRC_2022 + " " + SRC_2026, "neighborhood centroid"),
    ("Lapaz", -0.25098, 5.60553, SRC_2026, "neighborhood centroid"),
    ("Mallam", -0.27968, 5.57298, SRC_2026, "neighborhood centroid"),
    ("Dansoman/Glefe area", -0.25683, 5.56776, SRC_2022 + " (Dansoman listed among flooded areas)", "approximate neighborhood centroid"),
]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "lon", "lat", "source_description", "coordinate_quality"])
        w.writerows(HOTSPOTS)
    print(f"Wrote {OUT} with {len(HOTSPOTS)} documented hotspots")


if __name__ == "__main__":
    main()
