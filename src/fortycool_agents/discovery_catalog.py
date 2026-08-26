from __future__ import annotations

from .models import PublicSiteCandidate, SiteInput


PUBLIC_DATA_CENTER_CATALOG = [
    PublicSiteCandidate(
        id="digital-realty-acc5",
        operator="Digital Realty",
        address="44521 Hastings Drive, Ashburn, VA 20147",
        site=SiteInput(
            name="Digital Realty ACC5",
            latitude=39.0200887,
            longitude=-77.4632384,
        ),
        operator_source_url=(
            "https://www.digitalrealty.com/data-centers/americas/northern-virginia/acc5"
        ),
    ),
    PublicSiteCandidate(
        id="digital-realty-iad24",
        operator="Digital Realty",
        address="43830 Devin Shafron Drive, Ashburn, VA 20147",
        site=SiteInput(
            name="Digital Realty IAD24",
            latitude=39.0043336,
            longitude=-77.4843499,
        ),
        operator_source_url=(
            "https://www.digitalrealty.com/data-centers/americas/northern-virginia/iad24"
        ),
    ),
    PublicSiteCandidate(
        id="equinix-dc14",
        operator="Equinix",
        address="7400 Infantry Ridge Road, Manassas, VA 20109",
        site=SiteInput(
            name="Equinix DC14",
            latitude=38.8034063,
            longitude=-77.5105681,
        ),
        operator_source_url=(
            "https://www.equinix.com/data-centers/americas-colocation/"
            "united-states-colocation/washington-dc-data-centers/dc14"
        ),
    ),
    PublicSiteCandidate(
        id="digital-realty-va3",
        operator="Digital Realty",
        address="1780 Business Center Drive, Reston, VA 20190",
        site=SiteInput(
            name="Digital Realty VA3",
            latitude=38.9487713,
            longitude=-77.3277043,
        ),
        operator_source_url=(
            "https://www.digitalrealty.com/data-centers/americas/northern-virginia/va3"
        ),
    ),
]


def public_catalog() -> list[PublicSiteCandidate]:
    return [candidate.model_copy(deep=True) for candidate in PUBLIC_DATA_CENTER_CATALOG]
