package org.albacete.simd.mAnDE;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Objects;
import weka.core.Instance;
import weka.core.Utils;

/**
 *
 * @author Pablo Torrijos Arenas
 */
public class mSP1DE {

    /**
     * Nombre del Super-Padre del mSP1DE.
     */
    private final String xi_s;

    /**
     * ID del Super-Padre del mSP1DE.
     */
    private final int xi_i;

    /**
     * Enlace el nombre de los hijos del mSP1DE con su tabla de probabilidad.
     */
    private final HashMap<String, double[][][]> hijos;

    /**
     * Lista de los hijos del mSP1DE.
     */
    private final HashSet<String> listaHijos;

    /**
     * Tabla de probabilidad global del mSP1DE.
     */
    private double[][] probGlobal;

    /**
     * Constructor. Crea un mSP1DE pasándole como argumento el nombre de la variable xi que va a ser Super-Padre del resto de variables junto a la clase 'y'.
     *
     * @param xi Variable padre
     */
    public mSP1DE(String xi) {
        this.xi_s = xi;
        this.xi_i = mAnDE.nToI.get(xi);
        this.listaHijos = new HashSet<>();
        this.hijos = new HashMap<>();
    }

    /**
     * Crea las tablas de probabilidades para el mSP1DE, tanto la global P(y,Xi) como la condicional de cada variable P(Xj|y,Xi)
     */
    protected void creaTablas() {
        this.probGlobal = new double[mAnDE.classNumValues] //y
                [mAnDE.varNumValues[xi_i]]; //Xi

        listaHijos.forEach((hijo) -> {
            this.hijos.put(hijo, new double[mAnDE.classNumValues] //y
                    [mAnDE.varNumValues[xi_i]] //Xi
                    [mAnDE.varNumValues[mAnDE.nToI.get(hijo)]]); //Xj
        });

        // Creación de las tablas de contingencia
        for (int i = 0; i < mAnDE.numInstances; i++) {
            Instance inst = mAnDE.data.get(i);

            // Creación de la tabla de probabilidad P(y,Xi)
            probGlobal[(int) inst.value(mAnDE.y)][(int) inst.value(xi_i)] += 1;

            // Creación de la tabla de probabilidad P(Xj|y,Xi)
            hijos.forEach((String xj, double[][][] tablaXj) -> {
                int xj_i = mAnDE.nToI.get(xj);
                tablaXj[(int) inst.value(mAnDE.y)][(int) inst.value(xi_i)][(int) inst.value(xj_i)] += 1;
            });
        }

        // Conversión a Distribución de Probabilidad Conjunta
        for (double[] probGlobal_y : probGlobal) {
            for (int j = 0; j < probGlobal_y.length; j++) {
                probGlobal_y[j] /= mAnDE.numInstances;
            }
        }

        // Conversión a Distribución de Probabilidad Condicional
        hijos.forEach((String xj, double[][][] tablaXj) -> {
            int xj_i = mAnDE.nToI.get(xj);
            double suma;
            for (double[][] tablaXj_y : tablaXj) {
                for (double[] tablaXj_y_xi : tablaXj_y) {
                    suma = Utils.sum(tablaXj_y_xi);
                    if (suma != 0) {
                        for (int k = 0; k < tablaXj_y_xi.length; k++) {
                            tablaXj_y_xi[k] /= suma;
                        }
                    }
                }
            }
        });
    }

    /**
     * Calcula las probabilidades para cada valor de la clase dada una instancia. Para ello, se aplica la fórmula: P(y,Xi) * (\prod_{i=1}^{Nhijos} P(Xj|y,Xi)), siendo Xi la variable padre en el mSP1DE, y Xj cada una de las variables hijas.
     *
     * @param inst Instancia sobre la que calcular la clase.
     * @return Probabilidades para cada valor de la clase plara la instancia dada.
     */
    protected double[] probabilidadesParaInstancia(Instance inst) {
        double[] res = new double[mAnDE.classNumValues];
        double xi = inst.value(xi_i);

        // Inicializamos la probabilidad de cada valor de la clase a P(y,xi)
        for (int i = 0; i < res.length; i++) {
            res[i] = probGlobal[i][(int) xi];
        }

        /* Para cada hijo Xj, multiplicamos P(Xj|y,Xi) por el resultado 
         * acumulado para cada uno de los valores de la clase
         */
        hijos.forEach((String xj_s, double[][][] tablaXj) -> {
            for (int i = 0; i < res.length; i++) {
                res[i] *= tablaXj[i][(int) xi][(int) inst.value(mAnDE.nToI.get(xj_s))];
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
     * Añade una variable como hijo en el mSP1DE.
     *
     * @param hijo Nombre de la variable a añadir como hijo.
     */
    protected void masHijos(String hijo) {
        if (!hijo.equals("")) {
            listaHijos.add(hijo);
        }
    }

    /**
     * Devuelve el número de hijos del mSP1DE.
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
        if (!(o instanceof mSP1DE)) {
            return false;
        }

        mSP1DE that = (mSP1DE) o;
        return super.equals(that)
                && Objects.equals(this.xi_i, that.xi_i)
                && Objects.equals(this.xi_s, that.xi_s);
    }

    @Override
    public int hashCode() {
        int hash = 3;
        hash = 47 * hash + Objects.hashCode(this.xi_i);
        return hash;
    }
}
