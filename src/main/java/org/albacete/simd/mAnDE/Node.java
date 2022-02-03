package org.albacete.simd.mAnDE;

import java.util.ArrayList;
import java.util.HashMap;

/**
 *
 * @author Pablo Torrijos Arenas
 */
public class Node {

    private String id;
    private String nombre;
    private Node padre;
    private HashMap<String, Node> hijos;

    /**
     * Constructor. Crea un nodo padre, pasándole el ID y el nombre.
     *
     * @param id ID del nodo
     * @param nombre Nombre del nodo
     */
    public Node(String id, String nombre) {
        this.padre = this;
        this.id = id;
        this.nombre = nombre;
        this.hijos = new HashMap();
    }

    /**
     * Constructor. Crea un nodo hijo, pasándole el ID y el nodo padre.
     *
     * @param id ID del nodo
     * @param padre Padre del nodo
     */
    public Node(String id, Node padre) {
        this.padre = padre;
        this.id = id;
        this.nombre = "";
        this.hijos = new HashMap();
    }

    /**
     * Añade un hijo.
     *
     * @param hijo Hijo a añadir
     */
    protected void addHijo(Node hijo) {
        this.hijos.put(hijo.getId(), hijo);
    }

    /**
     * Devuelve el array de hijos.
     *
     * @return El array de hijos
     */
    protected ArrayList<String> getHijosArray() {
        ArrayList<String> temp = new ArrayList();
        hijos.values().forEach((hijo) -> {
            if (!hijo.nombre.equals("")) {
                temp.add(hijo.nombre);
            }
        });
        return temp;
    }

    /**
     * Devuelve el array de hijos que no sean "no".
     *
     * @param no El hijo que no se debe devolver
     * @return El array de hijos
     */
    protected ArrayList<String> getHijosArray(String no) {
        ArrayList<String> temp = new ArrayList();
        hijos.values().forEach((hijo) -> {
            if ((!hijo.nombre.equals(no)) && (!hijo.nombre.equals(""))) {
                temp.add(hijo.nombre);
            }
        });
        return temp;
    }

    /**
     * @return the id
     */
    public String getId() {
        return id;
    }

    /**
     * @param id the id to set
     */
    public void setId(String id) {
        this.id = id;
    }

    /**
     * @return the padre
     */
    public Node getPadre() {
        return padre;
    }

    /**
     * @param padre the padre to set
     */
    public void setPadre(Node padre) {
        this.padre = padre;
    }

    /**
     * @return the hijos
     */
    public HashMap<String, Node> getHijos() {
        return hijos;
    }

    /**
     * @param hijos the hijos to set
     */
    public void setHijos(HashMap<String, Node> hijos) {
        this.hijos = hijos;
    }

    /**
     * @return the nombre
     */
    public String getNombre() {
        return nombre;
    }

    /**
     * @param nombre the nombre to set
     */
    public void setNombre(String nombre) {
        this.nombre = nombre;
    }
}
