from astroquery.vizier import Vizier
catalog_list = Vizier.find_catalogs('J/ApJS/204/5')
Vizier.ROW_LIMIT = -1

cata = Vizier.get_catalogs(catalog_list.keys())[0]
print len(cata)