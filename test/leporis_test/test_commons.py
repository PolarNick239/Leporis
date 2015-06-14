__author__ = "Polyarnyi Nickolay"


import pkg_resources


test_resources_provider = pkg_resources.get_provider('resources')
resources_dir_path = test_resources_provider.get_resource_filename(__name__, '.')
