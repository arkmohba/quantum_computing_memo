from ngt import base as ngt
import random


def main():
    dim = 10
    objects = []
    for i in range(0, 100):
        vector = random.sample(range(100), dim)
        objects.append(vector)

    query = objects[0]
    # index = ngt.Index.create(b"tmp", dim)
    # index.insert(objects)
    # You can also insert objects from a file like this.
    # index.insert_from_tsv('list.tsv')

    # index.save()
    # You can load saved the index like this.
    index = ngt.Index(b"tmp")

    result = index.search(query, 3)

    for i, o in enumerate(result):
        print(str(i) + ": " + str(o.id) + ", " + str(o.distance))
        object = index.get_object(o.id)
        print(object)

if __name__ == "__main__":
    main()