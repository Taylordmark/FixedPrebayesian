class BDDCategory:
    """
    Stores information about Berkeley Deep Drive Categories, including superclass, subcategories, alias, and color

    Parameters:
    name:str = Name of the category, case insensitive,
    aliases:str = Name of alternative names which are also accepted
    color:str = Name of the color which the class is represented by (Default "black")
    """
    def __init__(self, name:str, aliases:list=[], color:str="black") -> None:
        self.name = name
        self.subcategories = {}
        self.color = color
        self.supercategory = None
        self.aliases = aliases

    def set_supercategory(self, superclass):
        """
        Sets a category's supercategory, as well as adding it to the other category's subcategory dictionary
        """
        self.supercategory = superclass
        superclass.subcategories[self.name] = self


    """
    Gets a subcategory by name, set recursive to true if you want to get any subcategory, otherwise it only gets from the immediate layer below
    Set use aliases to true if you want to include alternative names
    """
    def get_subcategory(self, name:str, use_aliases:bool=True, recursive:bool=True):

        for key in self.subcategories.keys():
            subc = self.subcategories[key]
            if(self.compare_names(name, subc, use_aliases)):
                return subc
            elif recursive:
                ret = self.__get_subcategory_recursive(name, subc, use_aliases)
                if (ret is not None):
                    return ret


    """
    Returns the supercategory if the value is of layer 2, returns the value if of layer 1
    """           
    def get_supercategory(self, name:str, use_aliases:bool=True):
        try :
            subc = self.get_subcategory(name, use_aliases, True)

            if (subc.name in self.subcategories.keys()):
                return subc
            else:
                return subc.supercategory
        except AttributeError:
            return None
        
    """
    Returns false if the subcategory is None
    """
    def has_subcategory(self, name:str) -> bool:
        return self.get_subcategory(name) is not None
    

    """
    Returns the "official" name of any alias
    """
    def proper_name(self, name:str) -> str:
        subc = self.get_subcategory(name)

        if (subc is not None):
            return subc.name
        else:
            return "Unknown"


    """
    Recursive function used to find subcategory, don't use on its own
    """
    def __get_subcategory_recursive(self, name:str, category, use_aliases:bool=True):
        for key in category.subcategories.keys():
            subc = category.subcategories[key]
            if(self.compare_names(name, subc, use_aliases)):
                return subc
            else:
                self.__get_subcategory_recursive(name, subc, use_aliases)


    """
    Returns true if the name is the same as the category's name.
    If uses aliases is true, this includes alternative names
    """
    def compare_names(self, name:str, category, use_aliases:bool=True):

        ret = str.lower(str(name)) == str.lower(str(category))

        if (use_aliases and not ret):
            for alias in category.aliases:
                if (str.lower(str(name)) == str.lower(alias)):
                    ret = True
                    break
        
        return ret


    """
    Returns a layer of the hierarchy as a list
    To get everything, set layer to the max depth and set get_all to true
    """
    def get_subclass_layer_as_list(self, layer:int = 1, node=None, return_list:list=[], get_all:bool=False):

        if (node is None):
            node = self

        if (layer == 0):
            return return_list.append(node)
        for key in node.subcategories.keys():
            print(key)

            if (layer == 1 or get_all):
                return_list.append(node.subcategories[key])
            if (layer > 1):
                self.get_subclass_layer_as_list(layer-1, node.subcategories[key], return_list)


    def subclass_list_layer2(self):
        ret = []
        lay1 = self.get_subclass_layer_as_list()

        for subc in lay1:
            ret.a

    def __str__(self) -> str:
        return self.name
    
    def __eq__(self, __value: object) -> bool:
        if (type(__value) is str):
            return self.compare_names(__value, self)
        return self.compare_names(__value.name, self)





BDD_CATEGORIES =  BDDCategory("Categories")

MOT_CATEGORIES = BDDCategory("Categories")

vehicle = BDDCategory("Vehicle", color="darkblue")
vehicle.set_supercategory(BDD_CATEGORIES)

person = BDDCategory("Person", color="darkred")
person.set_supercategory(BDD_CATEGORIES)

bike = BDDCategory("Bike", color="darkgreen")
bike.set_supercategory(BDD_CATEGORIES)

unknown = BDDCategory("Unknown", color="gray")
unknown.set_supercategory(BDD_CATEGORIES)

car = BDDCategory("Car", color="blue")
car.set_supercategory(vehicle)

truck = BDDCategory("Truck", ["Lorry", "Semi"], color="aquamarine")
truck.set_supercategory(vehicle)

train = BDDCategory("Train", ["Tram", "Subway", "Trolly"], color="violet")
train.set_supercategory(vehicle)

bus = BDDCategory("Bus", ["School bus"], color="indigo")
bus.set_supercategory(vehicle)

pedestrian = BDDCategory("Pedestrian", color="orange")
pedestrian.set_supercategory(person)

rider = BDDCategory("Rider", color="yellow")
rider.set_supercategory(person)


bicycle = BDDCategory("Bicycle")
bicycle.set_supercategory(bike)

bicycle = BDDCategory("Motorcycle")
bicycle.set_supercategory(bike)



#MOT CATEGORIES

person2 = BDDCategory("Person", color="darkred")
person2.set_supercategory(MOT_CATEGORIES)

vehicle2 = BDDCategory("Vehicle", color="darkblue")
vehicle2.set_supercategory(MOT_CATEGORIES)

misc = BDDCategory("Misc")
misc.set_supercategory(MOT_CATEGORIES)

unknown2 = BDDCategory("Unknown", color="gray")
unknown2.set_supercategory(MOT_CATEGORIES)

pedestrian2 = BDDCategory("Pedestrian", color="orange", aliases=["1"])
pedestrian2.set_supercategory(person2)

persononvehicle = BDDCategory("Person on Vehicle", color="yellow", aliases=["2", 
                                                                             "PersonOnVehicle", "Rider"])
persononvehicle.set_supercategory(person2)

car2 = BDDCategory("Car", color="blue", aliases=["3"])
car2.set_supercategory(vehicle2)

bicycle2 = BDDCategory("Bicycle", color="green", aliases=["4", "Bike"])
bicycle2.set_supercategory(vehicle2)

motorcycle2 = BDDCategory("Motorcycle", color="darkgreen", aliases=["5", "Motorbike"])
motorcycle2.set_supercategory(vehicle2)

nonmotor = BDDCategory("Nonmotorized Vehicle", color="purple", aliases=["6", "NonMotorizedVehicle"
                                                                     "Scooter", "Cart"])
nonmotor.set_supercategory(vehicle2)

staticperson = BDDCategory("Static Person", color="red", aliases=["7", "StaticPerson", "Onlooker"])
staticperson.set_supercategory(person2)

distractor = BDDCategory("Distractor", color="magenta", aliases=["8"])
distractor.set_supercategory(misc)

occluder = BDDCategory("Occluder", color="hotpink", aliases=["9"])
occluder.set_supercategory(misc)

occluderotg = BDDCategory("Occluder on the Ground", color="lavender", aliases=["10", "OccluderOnTheGround"])
occluderotg.set_supercategory(misc)

occluderf = BDDCategory("Occluder Full", color="pink", aliases=["11", "OccluderFull"])
occluderf.set_supercategory(misc)

reflection = BDDCategory("Reflection", color="violet", aliases=["12"])
reflection.set_supercategory(misc)




YOLO_CATEGORIES = BDDCategory("Categories")
person = BDDCategory("Person", aliases=["0"])
person.set_supercategory(YOLO_CATEGORIES)
car = BDDCategory("Car", aliases=["1"])
car.set_supercategory(YOLO_CATEGORIES)
motorcycle = BDDCategory("Motorcycle", aliases=["2"])

airplane = BDDCategory("Airplane", aliases=["3"])

bus = BDDCategory("Bus", aliases=["4"])

train = BDDCategory("Train", aliases=["5"])

truck = BDDCategory("Truck", aliases=["6"])

boat = BDDCategory("Boat", aliases=["7"])

traffic_light = BDDCategory("Traffic Light", aliases=["8"])

fire_hydrant = BDDCategory("Fire Hydrant", aliases=["9"])

street_sign = BDDCategory("Street Sign", aliases=["10"])

stop_sign = BDDCategory("Stop Sign", aliases=["11"])

parking_meter = BDDCategory("Parking Meter", aliases=["12"])

bench = BDDCategory("Bench", aliases=["13"])

cat = BDDCategory("Cat", aliases=["14"])

dog = BDDCategory("Dog", aliases=["15"])








