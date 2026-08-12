"""Generate bugs/bug_NN.py from baseline.py. Each is one targeted change.

Kept in the repo so you can see there is no magic: a bug file is the baseline
with a single edit. Do not read this before diagnosing -- it names them.
"""
import os, re, textwrap
base = open("baseline.py").read()
os.makedirs("bugs", exist_ok=True)

HEAD = ('"""bug_{n:02d} -- one fault. Diagnose from symptoms before reading.\n\n'
        '    python3 diagnose.py {n}\n"""\n')

EDITS = {
 1: [("            opt.zero_grad()                   # 1  gradients accumulate\n", "")],
 2: [("            opt.step()                        # 5\n", "")],
 3: [("            loss.backward()                   # 4\n", "")],
 4: [("        self.layers = nn.ModuleList()            # a list that is also a Module",
      "        self.layers = []")],
 5: [("        super().__init__()                       # required -- §8", "        pass")],
 6: [("        return self.head(self.drop(x))           # raw logits -- §25 of Part II",
      "        return torch.softmax(self.head(self.drop(x)), dim=1)")],
 7: [("    model.eval()\n    with torch.no_grad():", "    with torch.no_grad():")],
 8: [("            running += loss.item()            # .item() -- §3, or you leak",
      "            running = running + loss")],
 9: [("    loader = DataLoader(TensorDataset(Xtr, ytr),\n"
      "                        batch_size=batch_size, shuffle=True)",
      "    order = torch.argsort(ytr)            # sort by class\n"
      "    Xtr, ytr = Xtr[order], ytr[order]\n"
      "    loader = DataLoader(TensorDataset(Xtr, ytr),\n"
      "                        batch_size=batch_size, shuffle=False)")],
10: [("def train(epochs=8, batch_size=64, lr=1e-3, seed=0, verbose=True):",
      "def train(epochs=8, batch_size=64, lr=1.0, seed=0, verbose=True):")],
11: [("    X, y = make_data()", "    X, y = make_data()\n    X = X.double()")],
}

for n, edits in EDITS.items():
    src = base
    for old, new in edits:
        assert old in src, f"bug {n}: anchor not found"
        src = src.replace(old, new, 1)
    src = src.replace('"""\nbrokenrepo — the correct baseline. Stage 1 of the Part IV project.',
                      '"""\nbug_%02d' % n)
    src = re.sub(r'^"""\nbug_(\d+).*?"""', HEAD.format(n=n).rstrip(), src,
                 count=1, flags=re.S)
    src = src.replace('if __name__ == "__main__":',
                      'if __name__ == "__main__":')
    open(f"bugs/bug_{n:02d}.py", "w").write(src)
print("wrote", len(EDITS), "bug files")
