package org.albacete.simd.mAnDE;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Objects;
import weka.core.Instance;
import weka.core.Utils;

/**
 *
 * @author Pablo Torrijos Arenas
 */
public class mSP2DE {

    /**
     * Nombre del primer Super-Padre del mSP2DE.
     */
    private final String xi1_s;

    /**
     * ID del primer Super-Padre del mSP2DE.
     */
    private final int xi1_i;

    /**
     * Nombre del segundo Super-Padre del mSP2DE.
     */
    private final String xi2_s;

    /**
     * ID del segundo Super-Padre del mSP2DE.
     */
    private final int xi2_i;

    /**
     * Enlace el nombre de los hijos del mSP2DE con su tabla de probabilidad.
     */
    private final HashMap<String, double[][][][]> hijos;

    /**
     * Lista de los hijos del mSP2DE.
     */
    private final HashSet<String> listaHijos;

    /**
     * Tabla de probabilidad global del mSP2DE.
     */
    private double[][][] probGlobal;

    /**
     * Constructor. Crea un mSP2DE pasándole como argumento el nombre de las dos variable xi1 y xi2 que van a ser Super-Padres del resto de variables junto a la clase 'y'.
     *
     * @param xi1 Variable Padre 1
     * @param xi2 Variable Padre 2
     */
    public mSP2DE(String xi1, String xi2) {
        this.xi1_s = xi1;
        this.xi2_s = xi2;
        this.xi1_i = mAnDE.nToI.get(xi1);
        this.xi2_i = mAnDE.nToI.get(xi2);
        this.listaHijos = new HashSet<>();
        this.hijos = new HashMap<>();
    }

    /**
     * Crea las tablas de probabilidades para el mSP2DE, tanto la global P(y,Xi) como la condicional de cada variable P(Xj|y,Xi)
     */
    protected void creaTablas() {
        this.probGlobal = new double[mAnDE.classNumValues] //y
                [mAnDE.varNumValues[xi1_i]] //Xi1
                [mAnDE.varNumValues[xi2_i]];    //Xi2

        listaHijos.forEach((hijo) -> {
            this.hijos.put(hijo, new double[mAnDE.classNumValues] //y
                    [mAnDE.varNumValues[xi1_i]] //Xi1
                    [mAnDE.varNumValues[xi2_i]] //Xi2
                    [mAnDE.varNumValues[mAnDE.nToI.get(hijo)]]); //Xj
        });

        // Creación de las tablas de contingencia
        for (int i = 0; i < mAnDE.numInstances; i++) {
            Instance inst = mAnDE.data.get(i);

            // Creación de la tabla de probabilidad P(y,Xi1,Xi2)
            probGlobal[(int) inst.value(mAnDE.y)][(int) inst.value(xi1_i)][(int) inst.value(xi2_i)] += 1;

            // Creación de la tabla de probabilidad P(Xj|y,Xi1,Xi2)
            hijos.forEach((String xj, double[][][][] tablaXj) -> {
                int xj_i = mAnDE.nToI.get(xj);
                tablaXj[(int) inst.value(mAnDE.y)][(int) inst.value(xi1_i)][(int) inst.value(xi2_i)][(int) inst.value(xj_i)] += 1;
            });
        }

        // Conversión a Distribución de Probabilidad Conjunta
        for (double[][] probGlobal_y : probGlobal) {
            for (double[] probGlobal_y_x1 : probGlobal_y) {
                for (int j = 0; j < probGlobal_y_x1.length; j++) {
                    probGlobal_y_x1[j] /= mAnDE.numInstances;
                }
            }
        }

        // Conversión a Distribución de Probabilidad Condicional
        hijos.forEach((String xj, double[][][][] tablaXj) -> {
            int xj_i = mAnDE.nToI.get(xj);
            double suma;
            for (double[][][] tablaXj_y : tablaXj) {
                for (double[][] tablaXj_y_xi1 : tablaXj_y) {
                    for (double[] tablaXj_y_xi1_xi2 : tablaXj_y_xi1) {
                        suma = Utils.sum(tablaXj_y_xi1_xi2);
                        if (suma != 0) {
                            for (int k = 0; k < tablaXj_y_xi1_xi2.length; k++) {
                                tablaXj_y_xi1_xi2[k] /= suma;
                            }
                        }
                    }
                }
            }
        });
    }

    /**
     * Calcula las probabilidades para cada valor de la clase dada una instancia. Para ello, se aplica la fórmula: P(y,Xi1,Xi2) * (\prod_{i=1}^{Nhijos} P(Xj|y,Xi1,Xi2)), siendo Xi1 e Xi2 las variable padres en el mSP2DE, y Xj cada una de las variables hijas.
     *
     * @param inst Instancia sobre la que calcular la clase.
     * @return Probabilidades para cada valor de la clase plara la instancia dada.
     */
    protected double[] probabilidadesParaInstancia(Instance inst) {
        double[] res = new double[mAnDE.classNumValues];
        double xi1 = inst.value(xi1_i);
        double xi2 = inst.value(xi2_i);

        // Inicializamos la probabilidad de cada valor de la clase a P(y,xi)
        for (int i = 0; i < res.length; i++) {
            res[i] = probGlobal[i][(int) xi1][(int) xi2];
        }

        /* Para cada hijo Xj, multiplicamos P(Xj|y,Xi1,Xi2) por el resultado 
         * acumulado para cada uno de los valores de la clase
         */
        hijos.forEach((String xj_s, double[][][][] tablaXj) -> {
            for (int i = 0; i < res.length; i++) {
                res[i] *= tablaXj[i][(int) xi1][(int) xi2][(int) inst.value(mAnDE.nToI.get(xj_s))];
            }
        });

        // Normalizamos los valores dividiéndolos entre la suma de todos ellos
        double suma = Utils.sum(res);
        if (suma != 0) {
            for (int i = 0; i < res.length; i++) {
                res[i] /= suma;
            }
        }

        return res;
    }

    /**
     * Añade una variable como hijo en el mSP2DE.
     *
     * @param hijo Nombre de la variable a añadir como hijo.
     */
    protected void masHijos(String hijo) {
        if (!hijo.equals("")) {
            listaHijos.add(hijo);
        }
    }

    /**
     * Añade varias variables como hijos en el mSP2DE.
     *
     * @param hijos Nombre de las variables a añadir como hijos.
     */
    protected void masHijos(ArrayList<String> hijos) {
        hijos.forEach((hijo) -> {
            if (!hijo.equals("")) {
                listaHijos.add(hijo);
            }
        });
    }

    /**
     * Devuelve el número de hijos del mSP2DE.
     */
    protected int getNHijos() {
        return hijos.size();
    }

    /**
     *
     * @param o Objeto a comparar.
     * @return True si los objetos son iguales y False si no lo son.
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null) {
            return false;
        }
        if (!(o instanceof mSP2DE)) {
            return false;
        }

        mSP2DE that = (mSP2DE) o;
        return super.equals(that)
                && ((Objects.equals(this.xi1_i, that.xi1_i)
                && Objects.equals(this.xi1_s, that.xi1_s)
                && Objects.equals(this.xi2_i, that.xi2_i)
                && Objects.equals(this.xi2_s, that.xi2_s))
                || (Objects.equals(this.xi1_i, that.xi2_i)
                && Objects.equals(this.xi1_s, that.xi2_s)
                && Objects.equals(this.xi2_i, that.xi1_i)
                && Objects.equals(this.xi2_s, that.xi1_s)));
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 19 * hash + Objects.hashCode(this.xi1_s);
        hash = 19 * hash + this.xi1_i;
        hash = 19 * hash + Objects.hashCode(this.xi2_s);
        hash = 19 * hash + this.xi2_i;
        return hash;
    }

}
