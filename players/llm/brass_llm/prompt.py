"""Versioned seat instructions and lossless public history representation."""

import json
from functools import lru_cache

import brass

from .protocol import compact

PROTOCOL_VERSION = "brass-llm-turn/1"
PROMPT_VERSION = "birmingham-4p/1"
RULES = """Play four-player Brass: Birmingham. Aim for unique first after final VP, then income level, then money.
The Canal and Rail eras each have eight rounds. Players normally take two actions per turn (one in the first Canal round).
Least spending this round acts earliest next round. Income level determines round income; shortfalls can force tile removal.
At each era end, flipped industries score printed VP. Links score all adjacent flipped-tile link icons and printed merchant icons. Level I industries and canals are cleared after Canal; surviving flipped industries can score again.
Build places the lowest available level of that industry, paying its money, coal and iron costs. A location card builds there, an industry card normally builds within your network. Network actions extend your links. Develop discards low tiles to reach higher levels. Loan gives £30 and drops income three levels. Scout exchanges three discards for one of each wild. Pass discards a card.
Connected closest mines supply coal, then the market through a merchant connection. Iron can come from any ironworks, then the market. Empty mines, ironworks and breweries flip and advance income. Selling flips a connected matching industry and consumes its beer requirement. Your own brewery can supply beer anywhere; opponents' breweries need a connection. The selected merchant's beer may pay for a sale and trigger its bonus.
Follow-up phases can offer a second rail (extra coal and brewery beer), another development, another sale, a free bonus development, or shortfall tile removal. EndAction finishes an optional continuation.
Component arrays use native zero-based IDs. Location IDs identify named towns, slots index industries, links connect locations, merchant tile IDs begin at each first_tile. mat_remaining rows follow industries, columns start at level I. income_levels maps income spaces to levels. from_tiles entries are [slot,count]. Market quantities are explicit.
The initial_public snapshot and ordered history describe every public decision. For each event apply its changes in order: [path,value] replaces a field or list element reached by the keys/indices in path.
You see your own hand and current public board. The actions array is the entire current legal list. Choose its zero-based index, including the exact discard and resource sources. Respond only with a JSON object with exactly turn_id (the supplied string) and action_id (an integer index)."""


@lru_cache(maxsize=1)
def instructions():
    data = json.loads(brass.component_data())
    # Static card multiset is public. Never serialize an ordered deck, even as a reference example.
    deck = data.pop("deck_4p")
    counts = {}
    for card in deck:
        key = compact(card)
        counts[key] = counts.get(key, 0) + 1
    data["card_counts"] = [[json.loads(key), count] for key, count in counts.items()]
    return RULES + "\nComponents " + compact(data)


def changes(old, new, path=()):
    """Lossless structural replacements. Only changed subtrees cross the wire."""
    if old == new:
        return
    if isinstance(old, dict) and isinstance(new, dict) and old.keys() == new.keys():
        for key in new:
            yield from changes(old[key], new[key], path + (key,))
    elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        for i, value in enumerate(new):
            yield from changes(old[i], value, path + (i,))
    else:
        yield [path, new]
