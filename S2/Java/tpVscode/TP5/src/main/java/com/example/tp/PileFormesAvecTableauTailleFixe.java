package com.example.tp;

import java.util.Arrays;
import java.util.StringJoiner;

public class PileFormesAvecTableauTailleFixe implements PileFormes {
    private Forme[] formes;
    private int sommet;

    public PileFormesAvecTableauTailleFixe(int taille) {
        formes = new Forme[taille];
        sommet = 0;
    }

    public PileFormesAvecTableauTailleFixe(Forme[] formes, int taille) {
        this.formes = Arrays.copyOf(formes, taille > formes.length ? taille : formes.length);
        sommet = formes.length;
    }

    @Override
    public void empiler(Forme f) {
        if (sommet != formes.length)
            formes[sommet++] = f;
    }

    @Override
    public void depiler() {
        formes[--sommet] = null;
    }

    @Override
    public Forme sommet() {
        if (sommet == 0)
            throw new IndexOutOfBoundsException();
        return formes[sommet - 1];
    }

    @Override
    public boolean vide() {
        return sommet == 0;
    }

    @Override
    public String toString() {
        StringJoiner s = new StringJoiner(", ");
        for (var i = 0; i < sommet; i++) {
            s.add(formes[i].toString());
        }
        return "[ " + s + " ]";
    }

}