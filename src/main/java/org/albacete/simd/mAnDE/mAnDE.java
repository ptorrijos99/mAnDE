/**
 *
 * @author Pablo Torrijos Arenas
 */
package org.albacete.simd.mAnDE;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Vector;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.REPTree;
import weka.core.Capabilities;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;

import static weka.classifiers.AbstractClassifier.runClassifier;
import weka.classifiers.bayes.BayesNet;
import weka.core.Option;

public class mAnDE extends AbstractClassifier implements
        OptionHandler {

    /**
     * Para serialización.
     */
    private static final long serialVersionUID = 3545430914549890589L;

    // Variables auxialiares //
    /**
     * Instancias.
     */
    protected static Instances data;

    /**
     * El filtro de discretización.
     */
    protected weka.filters.supervised.attribute.Discretize discretizador = null;

    /**
     * Filtro de discretización para "discretizar4".
     */
    protected Discretize discretizadorNS = null;

    /**
     * HashMap que contiene a los mSPnDEs.
     */
    private HashMap<Integer, Object> mSPnDEs;

    /**
     * HashMap para convertir de nombre de variable a índice.
     */
    public static HashMap<String, Integer> nToI;

    /**
     * HashMap para convertir de índice de variable a nombre.
     */
    public static HashMap<Integer, String> iToN;

    /**
     * Número de valores por variable.
     */
    public static int[] varNumValues;

    /**
     * Número de valores de la clase.
     */
    public static int classNumValues;

    /**
     * Índice de la clase.
     */
    public static int y;

    /**
     * Número de instancias.
     */
    public static int numInstances;

    /**
     * Variable para comprobar si algún valor es cero.
     */
    private static int[] ceros;

    /**
     * Naive Bayes para el modo NB.
     */
    private static NaiveBayes nb;

    /**
     * Indica si el modo Naive Bayes está activado.
     */
    private boolean modoNB = false;

    /**
     * Indica si se ha discretizado.
     */
    private boolean discretizado = false;

    // Parámetros de mAnDE //
    /**
     * Realiza árboles de Chow-Liu en lugar de árboles de decisión.
     */
    private boolean chowLiu = false;

    /**
     * Realiza árboles REPTree en lugar de J48.
     */
    private boolean repTree = false;

    /**
     * Poda los árboles de decisión.
     */
    private boolean poda = true;

    /**
     * Completa mAnDE con los mSPnDEs que no se hayan añadido.
     */
    private boolean completa = false;

    /**
     * Ejecuta un árboles, y para cada variable del mismo un nuevo árbol en el que dicha variable actúe como clase.
     */
    private boolean variosArboles = false;

    /**
     * Ejecuta un ensemble de árboles de decisión en lugar de un solo árbol.
     */
    private boolean ensemble = false;

    /**
     * Ejecuta un Random Forest en lugar de Bagging.
     */
    private boolean randomForest = true;

    /**
     * Porcentaje de instancias que se utilizarán para realizar cada árbol del ensemble.
     */
    private double bagSize = 100;

    /**
     * n del mAnDE.
     */
    private int n = 1;

    /**
     * Discretiza antes de ejecutar el árbol.
     */
    private boolean discretizarAntes = true;

    /**
     * Discretiza en 4 intervalos las variables no discretizadas.
     */
    private boolean discretizar4 = false;

    /**
     * Usa mayoría en lugar de suma de probabilidades.
     */
    private boolean mayoria = false;

    /**
     * Crea la estructura del clasificador teniendo en cuenta los parámetros establecidos.
     *
     * @param instances Instancias a clasificar.
     * @throws java.lang.Exception
     */
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        // ¿El clasificador puede trabajar con esos datos?
        getCapabilities().testWithFail(instances);

        // Eliminar instancias sin clase
        instances.deleteWithMissingClass();

        // Comprobamos si tenemos que discretizar ahora o más tarde
        if (isDiscretizarAntes()) {
            data = new Instances(instances);
            discretizar(instances);
            // Liberamos el espacio de los datos por parámetro
            instances.delete();
        } else {
            data = instances;
        }

        // Inicializamos los diccionarios de índices
        inicializaNomToIndex();

        // Creamos los mSPnDE's
        try {
            crea_mSPnDEs();
        } catch (Exception ex) {
        }

        // Si no hemos creado mSPnDE's
        if (mSPnDEs.isEmpty()) {
            boolean empieza1 = false;
            if (!isEnsemble()) {
                empieza1 = true;
            }
            while (true) {
                if (!isEnsemble() && !empieza1) {
                    // Si no podemos ejecutar de ninguna manera, ejecutamos Naive Bayes
                    System.out.println("NO SE PUEDE EJECUTAR. EJECUTAMOS NAIVE BAYES.");
                    modoNB = true;
                    nb = new NaiveBayes();
                    nb.buildClassifier(data);
                    break;
                }
                if (!isEnsemble()) {
                    setEnsemble(true);
                    setRandomForest(true);
                } else if (isEnsemble()) {
                    bagSize *= 5;
                    if (getBagSize() > 100) {
                        if (isRandomForest()) {
                            setRandomForest(false);
                            bagSize = 100;
                        } else {
                            setEnsemble(false);
                            empieza1 = false;
                        }
                    }
                }
                try {
                    crea_mSPnDEs();
                } catch (Exception ex) {
                }

                // Si hemos creado mSPnDEs, terminamos
                if (!mSPnDEs.isEmpty()) {
                    break;
                }
            }
        }
        System.out.println("  mSPnDEs creados");

        // Discretizamos si no lo hemos hecho al principio
        if (!isDiscretizarAntes()) {
            discretizar(instances);
            // Liberamos el espacio de los datos por parámetro
            instances.delete();
        }

        // Si no hemos ejecutado Naive Bayes, calculamos las tablas de mAnDE
        if (!modoNB) {
            // Definimos variables globales
            y = data.classIndex();
            classNumValues = data.classAttribute().numValues();
            varNumValues = new int[data.numAttributes()];
            for (int i = 0; i < varNumValues.length; i++) {
                varNumValues[i] = data.attribute(i).numValues();
            }
            numInstances = data.numInstances();

            calculaTablas_mSPnDEs();
        }

        // Liberamos el espacio de los datos discretizados
        data.delete();

        nSPnDEsyVariables();
    }

    /**
     * Calcula las probabilidades de pertenencia a la clase para la Instancia de test proporcionada.
     *
     * @param instance Instancia a clasificar.
     * @return Distribución de probabilidad de pertenencia a la clase predicha.
     * @throws java.lang.Exception
     */
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] res = new double[classNumValues];

        final Instance instance_d;
        // Si hemos discretizado alguna instancia con el discretizador No Supervisado
        if (isDiscretizar4() && isDiscretizado() && (ceros != null)) {
            discretizadorNS.input(instance);
            discretizador.input(discretizadorNS.output());
            instance_d = discretizador.output();
        } else {
            discretizador.input(instance);
            instance_d = discretizador.output();
        }

        if (modoNB) {
            return nb.distributionForInstance(instance_d);
        }

        // Suma todas las probabilidades de los mSPnDE's
        mSPnDEs.forEach((id, spode) -> {
            double[] temp = new double[0];
            if (getN() == 1) {
                temp = ((mSP1DE) spode).probabilidadesParaInstancia(instance_d);
            } else if (getN() == 2) {
                temp = ((mSP2DE) spode).probabilidadesParaInstancia(instance_d);
            }

            if (isMayoria()) {
                res[Utils.maxIndex(temp)]++;
            } else {
                for (int i = 0; i < res.length; i++) {
                    res[i] += temp[i];
                }
            }
        });

        /* Normaliza el resultado. Si la suma es 0 (Utils.normalize nos devolverá
         * un IllegalArgumentException), establecemos el mismo valor en cada 
         * posible valor de la clase.
         */
        try {
            Utils.normalize(res);
        } catch (IllegalArgumentException ex) {
            for (int i = 0; i < res.length; i++) {
                res[i] = 1.0 / (varNumValues.length - 1);
            }
        }

        return res;
    }

    /**
     * Función que realiza la discretización, comprobando el parámetro "discretizar4" para realizar dicha discretización o no.
     *
     * @param instances Instancias a discretizar
     * @throws Exception
     */
    private void discretizar(Instances instances) throws Exception {
        // Si vamos a discretizar en 4 trozos las variables que se queden
        // enteras en un intervalo
        if (isDiscretizar4()) {
            // Discretizar instancias si es necesario
            discretizador = new weka.filters.supervised.attribute.Discretize();
            discretizador.setInputFormat(instances);
            instances = weka.filters.Filter.useFilter(instances, discretizador);

            // Añadimos las variables con solo un intervalo
            ArrayList<Integer> temp = new ArrayList();
            for (int i = 0; i < instances.numAttributes(); i++) {
                if (instances.attribute(i).numValues() == 1) {
                    temp.add(i);
                }
            }

            // Si hay, tenemos que discretizar por igual anchura
            if (!temp.isEmpty()) {
                ceros = new int[temp.size()];
                for (int i = 0; i < ceros.length; i++) {
                    ceros[i] = temp.get(i);
                }

                // Primero discretizamos las variables por igual anchura (4 bins)
                discretizadorNS = new Discretize();
                discretizadorNS.setBins(4);
                discretizadorNS.setInputFormat(data);
                discretizadorNS.setAttributeIndicesArray(ceros);
                data = weka.filters.Filter.useFilter(data, discretizadorNS);
                setDiscretizado(true);

                // Y luego el resto de variables
                discretizador = new weka.filters.supervised.attribute.Discretize();
                discretizador.setInputFormat(data);
                data = weka.filters.Filter.useFilter(data, discretizador);
            } else {
                // Si no, directamente la discretización es "instances"
                data = new Instances(instances);
            }
        } // Si solo vamos a discretizar una vez
        else {
            discretizador = new weka.filters.supervised.attribute.Discretize();
            discretizador.setInputFormat(data);
            data = weka.filters.Filter.useFilter(data, discretizador);
        }
    }

    /**
     * Crea los mSPnDE's necesarios, ejecutando los árboles establecidos en las opciones..
     */
    private void crea_mSPnDEs() throws Exception {
        mSPnDEs = new HashMap<>();

        if (chowLiu) {
            usaChowLiu();
        } else {
            if (variosArboles) {
                variosArboles();
            } else {
                Classifier[] arboles;

                J48 j48 = new J48();
                Bagging bagging;
                REPTree repT = new REPTree();

                // Definimos si queremos poda o no
                if (!poda) {
                    j48.setUnpruned(true);
                    repT.setNoPruning(true);
                }

                String[] opciones = new String[2];
                opciones[0] = "-num-slots";
                opciones[1] = "0";

                if (ensemble) {
                    if (randomForest) {
                        System.out.println("Ejecuta Random Forest");
                        RandomForest rf = new RandomForest();
                        // Establecemos el número de hilos en paralelo a 0 (automáticos)
                        rf.setOptions(opciones);
                        rf.setNumIterations(10);
                        rf.setBagSizePercent(bagSize);
                        rf.buildClassifier(data);
                        arboles = rf.getClassifiers();
                        for (Classifier arbol : arboles) {
                            graphToSPnDE(arbolParser(arbol));
                        }
                    } else {
                        System.out.println("Ejecuta Bagging");
                        bagging = new Bagging();
                        if (repTree) {
                            bagging.setClassifier(repT);
                        } else {
                            bagging.setClassifier(j48);
                        }
                        // Establecemos el número de hilos en paralelo a 0 (automáticos)
                        bagging.setOptions(opciones);
                        bagging.setNumIterations(10);
                        bagging.setBagSizePercent(bagSize);
                        bagging.buildClassifier(data);
                        arboles = bagging.getClassifiers();
                        for (Classifier arbol : arboles) {
                            graphToSPnDE(arbolParser(arbol));
                        }
                    }
                } else {
                    if (repTree) {
                        repT.buildClassifier(data);
                        graphToSPnDE(arbolParser(repT));
                    } else {
                        j48.buildClassifier(data);
                        graphToSPnDE(arbolParser(j48));
                    }
                }
            }

            // Si queremos añadir mSP1DE's con las variables que no están en el árbol
            if (isCompleta()) {
                if (getN() == 1) {
                    completaTodos_mSP1DEs();
                } else {
                    System.out.println("Completa solo puede usarse con n = 1. Se procede a desactivarlo.");
                }
            }
        }
    }

    /**
     * Lee el clasificador pasado por parámetro y devuelve un HashMap<String,Node> con los datos del mismo.
     *
     * @param clasificador Clasificador a parsear
     */
    private HashMap<String, Node> arbolParser(Classifier clasificador) {
        String[] lines = new String[0];
        try {
            lines = ((Drawable) clasificador).graph().split("\r\n|\r|\n");
        } catch (Exception ex) {
        }

        HashMap<String, Node> nodos = new HashMap();
        int inicio = -1, fin = -1;

        for (int i = 0; i < lines.length; i++) {
            if (lines[i].contains(" [label")) {
                inicio = i;
                break;
            }
        }

        for (int i = 0; i < lines.length; i++) {
            if (lines[i].contains(" [label")) {
                fin = i;
            }
        }

        if (inicio != -1) {
            // Añadimos al padre
            //N0 [label="petallength" ]             J48
            //N58640b4a [label="1: petalwidth"]     RandomTree
            String id = lines[inicio].substring(0, lines[inicio].indexOf(" [label"));
            String nombre = "", finLabel;
            if (lines[inicio].contains(": ")) {
                // RandomTree y REPTree
                try {
                    nombre = lines[inicio].substring(lines[inicio].indexOf(": ") + 2, lines[inicio].indexOf("\"]"));
                } catch (Exception ex) {
                }
            } else {
                // J48
                try {
                    nombre = lines[inicio].substring(lines[inicio].indexOf("=\"") + 2, lines[inicio].indexOf("\" ]"));
                } catch (Exception ex) {
                }
            }

            if (!"".equals(nombre)) {
                if (!nToI.containsKey(nombre)) {
                    nombre = nombre.replace("\\", "");
                }
                Node padre = new Node(id, nombre);
                nodos.put(id, padre);
            }

            for (int i = inicio + 1; i < fin; i++) {
                //N0->N1 [label="= \'(-inf-2.6]\'"]                     J48
                //N529c603a->N6f5883ef [label=" = \'(5.45-5.75]\'"]     RandomTree y REPTree
                if (lines[i].contains("->")) {
                    String id1 = lines[i].substring(0, lines[i].indexOf("->"));
                    String id2 = lines[i].substring(lines[i].indexOf("->") + 2, lines[i].indexOf(" [label"));

                    if (nodos.get(id2) == null) {
                        nodos.put(id2, new Node(id2, nodos.get(id1)));
                    }
                    nodos.get(id1).addHijo(nodos.get(id2));
                } //N0 [label="petallength" ]                J48
                //N58640b4a [label="1: petalwidth"]        RandomTree y REPTree
                else if (!lines[i].contains(" (")) {
                    if (lines[i].contains("\" ]")) {
                        finLabel = "\" ]";
                    } else {
                        finLabel = "\"]";
                    }

                    try {
                        id = lines[i].substring(0, lines[i].indexOf(" [label"));
                    } catch (Exception ex) {
                        System.out.println(lines[i]);
                    }

                    if (lines[i].contains(" : ")) {
                        // RandomTree y REPTree
                        nombre = lines[i].substring(lines[i].indexOf(" : ") + 3, lines[i].indexOf(finLabel));
                    } else if (lines[i].contains(": ")) {
                        // RandomTree y REPTree
                        nombre = lines[i].substring(lines[i].indexOf(": ") + 2, lines[i].indexOf(finLabel));
                    } else {
                        // J48
                        nombre = lines[i].substring(lines[i].indexOf("=\"") + 2, lines[i].indexOf(finLabel));
                    }

                    if (!nToI.containsKey(nombre)) {
                        nombre = nombre.replace("\\", "");
                    }
                    nodos.get(id).setNombre(nombre);
                }
            }
        }
        return nodos;
    }

    /**
     * Convierte un HashMap<String,Node> a una representación de mAnDE.
     */
    private void graphToSPnDE(HashMap<String, Node> nodos) {
        if (getN() == 1) {
            nodos.values().forEach((nodo) -> {
                nodo.getHijos().values().forEach((hijo) -> {
                    if (!nodo.getNombre().equals("") && !hijo.getNombre().equals("")) {
                        toSP1DE(nodo.getNombre(), hijo.getNombre());
                    }
                });
            });
        } else if (getN() == 2) {
            nodos.values().forEach((nodo) -> {
                nodo.getHijos().values().forEach((hijo) -> {
                    if (!nodo.getNombre().equals("") && !hijo.getNombre().equals("")) {
                        toSP2DE(nodo.getNombre(), hijo.getNombre(), nodo.getPadre().getNombre(),
                                nodo.getHijosArray(hijo.getNombre()), hijo.getHijosArray());
                    }
                });
            });
        }

    }

    /**
     * Crea un mSP1DE con la variable 'padre' (si no existe ya), y le añade como dependencia la variable 'hijo', y viceversa.
     *
     * @param padre Nombre del padre en el mSP1DE.
     * @param hijo Nombre del hijo en el mSP1DE.
     */
    private void toSP1DE(String padre, String hijo) {
        if (!padre.equals(hijo)) {
            if (!mSPnDEs.containsKey(padre.hashCode())) {
                mSPnDEs.put(padre.hashCode(), new mSP1DE(padre));
            }
            if (!hijo.equals("")) {
                if (!mSPnDEs.containsKey(hijo.hashCode())) {
                    mSPnDEs.put(hijo.hashCode(), new mSP1DE(hijo));
                }
                try {
                    ((mSP1DE) mSPnDEs.get(padre.hashCode())).masHijos(hijo);
                    ((mSP1DE) mSPnDEs.get(hijo.hashCode())).masHijos(padre);
                } catch (NullPointerException ex) {
                }
            }
        }
    }

    /**
     * Crea un mSP2DE con la variable 'padre' (si no existe ya), y le añade como dependencia la variable 'hijo', y viceversa.
     *
     * @param padre Nombre del padre en el mSP2DE.
     * @param hijo Nombre del hijo en el mSP2DE.
     * @param abuelo Nombre del padre del padre en el mSP2DE.
     * @param hermanos Nombre de los otros hijos del padre el mSP2DE.
     * @param nietos Nombre de los hijos del hijo el mSP2DE.
     */
    private void toSP2DE(String padre, String hijo, String abuelo,
            ArrayList<String> hermanos, ArrayList<String> nietos) {
        if (!padre.equals(hijo)) {
            if (!mSPnDEs.containsKey(padre.hashCode() + hijo.hashCode())) {
                mSPnDEs.put(padre.hashCode() + hijo.hashCode(), new mSP2DE(padre, hijo));
            }
            try {
                mSP2DE elem = ((mSP2DE) mSPnDEs.get(padre.hashCode() + hijo.hashCode()));
                if (!abuelo.equals(padre)) {
                    elem.masHijos(abuelo);
                }
                elem.masHijos(hermanos);
                elem.masHijos(nietos);
            } catch (NullPointerException ex) {
            }
        }
    }

    /**
     * Ejecuta en paralelo las funciones 'creaTabla()' de cada mSPnDE, y termina cuando todos la hayan ejecutado.
     */
    private void calculaTablas_mSPnDEs() {
        List<Object> list = new ArrayList<>(mSPnDEs.values());

        //Crea el pool de hilos, uno para cada mSP1DE
        ExecutorService executor = Executors.newFixedThreadPool(mSPnDEs.size());

        //Llama a la función del mSP1DE que crea la tabla para cada mSP1DE
        list.forEach((spode) -> {
            executor.execute(() -> {
                if (getN() == 1) {
                    ((mSP1DE) spode).creaTablas();
                } else if (getN() == 2) {
                    ((mSP2DE) spode).creaTablas();
                }
            });
        });

        executor.shutdown();    //Deja de admitir llamadas 
        try {
            //Espera hasta que acaben todos (a los T días para)
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.DAYS);
        } catch (InterruptedException ex) {
        }
    }

    /**
     * Método de ejecución utilizando árboles de Chow-Liu en lugar de árboles de decisión.
     *
     * @throws Exception
     */
    private void usaChowLiu() throws Exception {
        // Creamos el clasificador TAN, el cual crea un árbol de Chow-Liu ampliado
        BayesNet net = new BayesNet();
        weka.classifiers.bayes.net.search.local.TAN tan = new weka.classifiers.bayes.net.search.local.TAN();
        net.setSearchAlgorithm(tan);
        net.buildClassifier(data);

        // Parseamos el árbol generado
        String[] lines = new String[0];
        try {
            lines = net.toString().split("\r\n|\r|\n");
        } catch (Exception ex) {
        }

        HashMap<String, Node> nodos = new HashMap();

        for (int i = 4; i < lines.length; i++) {
            //att_1(1): class att_15   TAN
            String[] actual = lines[i].split(" ");
            if (!"LogScore".equals(actual[0])) {
                // Si es 3, un nodo tiene como padre a otro nodo y la clase
                if (actual.length == 3) {
                    // Removemos la parte sobrante del primer nombre
                    actual[0] = actual[0].substring(0, lines[i].indexOf("("));

                    if (nodos.get(actual[2]) == null) {
                        nodos.put(actual[2], new Node(actual[2], actual[2]));
                    }
                    if (nodos.get(actual[0]) == null) {
                        nodos.put(actual[0], new Node(actual[0], actual[0]));
                    }
                    nodos.get(actual[0]).setPadre(nodos.get(actual[2]));
                    nodos.get(actual[2]).addHijo(nodos.get(actual[0]));
                }
            }
        }

        graphToSPnDE(nodos);
    }

    /**
     * Método que cambia la ejecución, ejecutando un árbol, y para las variables que aparezcan en el mismo, otro árbol en el cual éstas se comporten como la clase.
     */
    private void variosArboles() {
        HashSet<String> variables = new HashSet();
        J48 j48 = new J48();
        try {
            j48.buildClassifier(data);
        } catch (Exception ex) {
        }

        // Obtenemos el nombre de las variables que aparecen en el árbol
        arbolParser(j48).values().forEach((var) -> {
            if (!"".equals(var.getNombre())) {
                variables.add(var.getNombre());
            }
        });

        int classIndex = data.classIndex();

        // Para cada una, vamos a ejecutar el árbol
        variables.forEach((var) -> {
            J48 arbol = new J48();
            data.setClassIndex(nToI.get(var));
            try {
                arbol.buildClassifier(data);
            } catch (Exception ex) {
            }

            HashSet<String> nuevasVariables = new HashSet();
            // Obtenemos el nombre de las variables
            arbolParser(arbol).values().forEach((var2) -> {
                if (!"".equals(var2.getNombre())) {
                    nuevasVariables.add(var2.getNombre());
                }
            });

            // Si la clase está dentro del árbol creado
            if (nuevasVariables.contains(iToN.get(classIndex))) {
                if (!mSPnDEs.containsKey(var.hashCode())) {
                    mSPnDEs.put(var.hashCode(), new mSP1DE(var));
                }
                nuevasVariables.forEach((temp) -> {
                    if (!temp.equals(classIndex)) {
                        ((mSP1DE) mSPnDEs.get(var.hashCode())).masHijos(temp);
                    }
                });
            }
        });

        data.setClassIndex(classIndex);
    }

    /**
     * Añade los mSP1DE's que no se hayan creado después de parsear los árboles. Cada uno de ellos no tendrá hijos, por lo que su probabilidad será P(Xi|y).
     */
    private void completaTodos_mSP1DEs() {
        for (int i = 0; i < data.numAttributes(); i++) {
            if (!mSPnDEs.containsKey(iToN.get(i).hashCode())) {
                mSPnDEs.put(iToN.get(i).hashCode(), new mSP1DE(iToN.get(i)));
            }
        }
    }

    /**
     * Inicializa los HashMap para convertir índices a nombres.
     */
    private void inicializaNomToIndex() {
        nToI = new HashMap<>();
        iToN = new HashMap<>();
        for (int i = 0; i < data.numAttributes(); i++) {
            nToI.put(data.attribute(i).name(), i);
            iToN.put(i, data.attribute(i).name());
        }
    }

    /**
     * Devuelve el número de nSPnDEs y de Variables por nSPnDE.
     *
     * @return El número de nSPnDEs y de Variables por nSPnDE
     * @throws java.io.IOException
     */
    public double[] nSPnDEsyVariables() throws IOException {
        double[] res = new double[2];
        res[0] = mSPnDEs.size();
        res[1] = 0;
        mSPnDEs.values().forEach((spode) -> {
            if (getN() == 1) {
                res[1] += (((mSP1DE) spode).getNHijos() / res[0]);
            } else if (getN() == 2) {
                res[1] += (((mSP2DE) spode).getNHijos() / res[0]);
            }
        });

        File f = new File("temp.txt");
        FileWriter fichero = new FileWriter(f, true);
        PrintWriter pw = new PrintWriter(fichero, true);
        pw.println(res[0] + "," + res[1]);
        fichero.close();

        return res;
    }

    /**
     * @param chowLiu El hiperparámetro chowLiu a establecer
     */
    public void setChowLiu(boolean chowLiu) {
        this.chowLiu = chowLiu;
    }

    /**
     * @param poda El hiperparámetro poda a establecer
     */
    public void setPoda(boolean poda) {
        this.poda = poda;
    }

    /**
     * @param completa El hiperparámetro completa a establecer
     */
    public void setCompleta(boolean completa) {
        this.completa = completa;
    }

    /**
     * @param bagSize El hiperparámetro bagSize a establecer
     */
    public void setBagSize(double bagSize) {
        this.bagSize = bagSize;
    }

    /**
     * @param discretizarAntes El hiperparámetro discretizarAntes a establecer
     */
    public void setDiscretizarAntes(boolean discretizarAntes) {
        this.discretizarAntes = discretizarAntes;
    }

    /**
     * @param discretizar4 El hiperparámetro discretizar4 a establecer
     */
    public void setDiscretizar4(boolean discretizar4) {
        this.discretizar4 = discretizar4;
    }

    /**
     * @param n El hiperparámetro n a establecer
     */
    public void setN(int n) {
        if (n > 0 && n < 3) {
            this.n = n;
        }
    }

    /**
     * @param mayoria El hiperparámetro mayoria a establecer
     */
    public void setMayoria(boolean mayoria) {
        this.mayoria = mayoria;
    }

    /**
     * @param repTree El hiperparámetro repTree a establecer
     */
    public void setRepTree(boolean repTree) {
        this.repTree = repTree;
    }

    /**
     * @param variosArboles El hiperparámetro variosArboles a establecer
     */
    public void setVariosArboles(boolean variosArboles) {
        this.variosArboles = variosArboles;
    }

    /**
     * @param ensemble El hiperparámetro ensemble a establecer
     */
    public void setEnsemble(boolean ensemble) {
        this.ensemble = ensemble;
    }

    /**
     * @param randomForest El hiperparámetro randomForest a establecer
     */
    public void setRandomForest(boolean randomForest) {
        this.randomForest = randomForest;
    }

    /**
     * @param discretizado El hiperparámetro discretizado a establecer
     */
    public void setDiscretizado(boolean discretizado) {
        this.discretizado = discretizado;
    }

    /**
     * @return El hiperparámetro discretizado
     */
    public boolean isDiscretizado() {
        return discretizado;
    }

    /**
     * @return El hiperparámetro chowLiu
     */
    public boolean isChowLiu() {
        return chowLiu;
    }

    /**
     * @return El hiperparámetro repTree
     */
    public boolean isRepTree() {
        return repTree;
    }

    /**
     * @return El hiperparámetro poda
     */
    public boolean isPoda() {
        return poda;
    }

    /**
     * @return El hiperparámetro completa
     */
    public boolean isCompleta() {
        return completa;
    }

    /**
     * @return El hiperparámetro variosArboles
     */
    public boolean isVariosArboles() {
        return variosArboles;
    }

    /**
     * @return El hiperparámetro ensemble
     */
    public boolean isEnsemble() {
        return ensemble;
    }

    /**
     * @return El hiperparámetro randomForest
     */
    public boolean isRandomForest() {
        return randomForest;
    }

    /**
     * @return El hiperparámetro bagSize
     */
    public double getBagSize() {
        return bagSize;
    }

    /**
     * @return El hiperparámetro n
     */
    public int getN() {
        return n;
    }

    /**
     * @return El hiperparámetro discretizarAntes
     */
    public boolean isDiscretizarAntes() {
        return discretizarAntes;
    }

    /**
     * @return El hiperparámetro discretizar4
     */
    public boolean isDiscretizar4() {
        return discretizar4;
    }

    /**
     * @return El hiperparámetro mayoria
     */
    public boolean isMayoria() {
        return mayoria;
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return The capabilities of this classifier.
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        return result;
    }

    /**
     * Returns an enumeration describing the available options
     *
     * @return an enumeration of all the available options
     */
    @Override
    public Enumeration listOptions() {
        Vector newVector = new Vector(11);

        newVector.addElement(new Option("\tn del mAnDE (1 o 2, por defecto 1)\n", "N", 1, "-N <int>"));
        newVector.addElement(new Option("\tRealiza árboles de Chow-Liu en lugar de árboles de decisión\n", "CH", 0, "-CH"));
        newVector.addElement(new Option("\tUsa árboles REPTree en lugar de J48\n", "REP", 0, "-REP"));
        newVector.addElement(new Option("\tNO realiza poda de los árboles de decisión\n", "P", 0, "-P"));
        newVector.addElement(new Option("\tCompleta los mSP1DEs que no hubieran sido creados\n", "C", 0, "-C"));
        newVector.addElement(new Option("\tRealiza un primer árbol, y con las variables que aparezcan en él, uno para cada una (variosÁrboles)\n", "V", 0, "-V"));
        newVector.addElement(new Option("\tRealiza un ensemble de árboles de decisión\n", "E", 0, "-E"));
        newVector.addElement(new Option("\tRealiza el ensemble de árboles de decisión usando Random Forest\n", "RF", 0, "-RF"));
        newVector.addElement(new Option("\tEstablece el número de instancias usadas para crear cada árbol cuando se usan ensembles (0, 100]\n", "B", 100, "-B <double>"));
        newVector.addElement(new Option("\tDiscretiza después de realizar el árbol de decisión\n", "D", 0, "-D"));
        newVector.addElement(new Option("\tDiscretiza en 4 bins las variables que se queden sin discretizar\n", "D4", 0, "-D4"));
        newVector.addElement(new Option("\tUsa un voto por mayoría en lugar de por suma de probabilidades\n", "M", 0, "-M"));

        return newVector.elements();
    }

    /**
     * Convierte una lista de opciones a los hiperparámetros del algoritmo.
     *
     * @param options Opciones a convertir
     * @throws java.lang.Exception
     */
    @Override
    public void setOptions(String[] options) throws Exception {
        String N = Utils.getOption('N', options);
        if (N.length() != 0) {
            n = Integer.parseInt(N);
            if (n < 1 || n > 2) {
                n = 1;
            }
        } else {
            n = 1;
        }

        chowLiu = Utils.getFlag("CH", options);

        repTree = Utils.getFlag("REP", options);

        poda = !Utils.getFlag("P", options);

        completa = Utils.getFlag("C", options);

        variosArboles = Utils.getFlag("V", options);

        ensemble = Utils.getFlag("E", options);

        randomForest = Utils.getFlag("RF", options);

        String Bag = Utils.getOption('B', options);
        if (Bag.length() != 0) {
            bagSize = Integer.parseInt(Bag);
        } else {
            bagSize = 100;
        }

        discretizarAntes = !Utils.getFlag("D", options);

        discretizar4 = Utils.getFlag("D4", options);

        mayoria = Utils.getFlag("M", options);

        Utils.checkForRemainingOptions(options);

    }

    /**
     * Devuelve las opciones actuales del clasificador.
     *
     * @return Un array de strings que se pueda añadir a setOptions
     */
    @Override
    public String[] getOptions() {
        Vector result = new Vector();

        result.add("-N");
        result.add("" + n);

        if (chowLiu) {
            result.add("-CH");
        }

        if (repTree) {
            result.add("-REP");
        }

        if (!poda) {
            result.add("-P");
        }

        if (completa) {
            result.add("-C");
        }

        if (variosArboles) {
            result.add("-V");
        }

        if (ensemble) {
            result.add("-E");
        }

        if (randomForest) {
            result.add("-RF");
        }

        result.add("-B");
        result.add("" + bagSize);

        if (!discretizarAntes) {
            result.add("-D");
        }

        if (discretizar4) {
            result.add("-D4");
        }

        if (mayoria) {
            result.add("-M");
        }

        return (String[]) result.toArray(new String[result.size()]);
    }

    /**
     * Método principal para probar esta clase.
     *
     * @param args las opciones
     */
    public static void main(String[] args) {
        runClassifier(new mAnDE(), args);
    }

}
