* limits a bit the functionality of pydash typing wise
* `flatten_deep` stuff is very hard to type


## TODO
* Only solution I see for chaining for now is to make a script using AST to build
a class which will have all the pydash function as methods returning the types
wrapped in `Chain`
* `ListIterateeT` cannot use `Iterable` it seems as callable args are contravariant,
pyright cannot match it.
* Same for `DictIterateeT` and `Mapping`
