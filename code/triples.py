class Triples:
    def __init__(self, triples, ori_triples=None):
        self.triples = set(triples)
        self.triple_list = list(self.triples)
        self.triples_num = len(triples)

        self.heads = set([triple[0] for triple in self.triple_list])
        self.props = set([triple[1] for triple in self.triple_list])
        self.tails = set([triple[2] for triple in self.triple_list])
        self.ents = self.heads | self.tails

        self.prop_list = list(self.props)
        self.ent_list = list(self.ents)
        self.prop_list.sort()
        self.ent_list.sort()

        if ori_triples is None:
            self.ori_triples = None
        else:
            self.ori_triples = set(ori_triples)

        self._generate_related_ents()
        self._generate_triple_dict()
        self._generate_ht()

    def _generate_related_ents(self):
        self.out_related_ents_dict = dict()
        self.in_related_ents_dict = dict()
        for h, r, t in self.triple_list:
            out_related_ents = self.out_related_ents_dict.get(h, set())
            out_related_ents.add(t)
            self.out_related_ents_dict[h] = out_related_ents

            in_related_ents = self.in_related_ents_dict.get(t, set())
            in_related_ents.add(h)
            self.in_related_ents_dict[t] = in_related_ents

    def _generate_triple_dict(self):
        self.rt_dict, self.hr_dict = dict(), dict()
        for h, r, t in self.triple_list:
            rt_set = self.rt_dict.get(h, set())
            rt_set.add((r, t))
            self.rt_dict[h] = rt_set
            hr_set = self.hr_dict.get(t, set())
            hr_set.add((h, r))
            self.hr_dict[t] = hr_set

    def _generate_ht(self):
        self.ht = set()
        for h, r, t in self.triples:
            self.ht.add((h, t))

